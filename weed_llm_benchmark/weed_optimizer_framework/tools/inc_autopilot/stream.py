"""Stream mode: the lane ticker of a continuous campaign (docs/CONTINUOUS_LOOP.md 6).

campaign.tick hands a campaign whose config says `mode: "stream"` to
StreamRun; experiment-mode campaigns keep campaign._Run, byte for byte.

Three lanes, one item in flight per lane:

    DATA   IDLE -> DISCOVER (L15, lab, detached) | NAMES (L26) | PROBE (LP)
                -> COLLECT (L16 job, or L16L on the lab + L16S sync) -> INTAKE (L16I)
                -> ADMIT (L17) -> IDLE, or INTAKE again while the intake left images
                deferred (shard_pending: the next shard); WAIT_DATA when collection
                is exhausted (D29)
    TRAIN  IDLE -> SEGMENT (L18, the experiment runs) -> COMMIT (L19) -> IDLE; L22 forks;
                V3FAM_DETR (L23K, D6's RF-DETR smoke, then its three seeds)
    MAINT  IDLE -> R0 (L23 splits, L23B baselines, LV verdicts, L25 Stage A, LI init,
                LA choose-arm, L28 Stage C) | NATIVE (L23N, a done measurement
                arm's native-resolution rescore) | BASE3 (L23V, E1's base v3
                build) | AGNOSTIC (L23E, E1's agnostic rescore and verdict)
                | E2_VERDICT (L23C, E2's rescore and verdict)
                | E2_ATTR (L23D, E2-C's attribution rescore and record)
                | ZOO (L23Z, the model-zoo audit: one chain of five jobs)
                | E3_SCORE (L23F, one E3 arm's two-stage dev scores)
                | E3_VERDICT (L23G, E3's verdict)
                | V3FAM_DATA (L23H) | V3FAM_BUILD (L23I) | V3FAM_SCORE (L23J)
                | V3FAM_VERDICT (L23L): the v3 experiment family, record only
                | DS1_DATA (L30) | DS1_SHEETS (L31, the lab) | DS1_BUILD (L32) | DS1_SCORE (L33)
                | DS1_VERDICT (L34): DS1, record only (DS1_LONG, L32X, on TRAIN)
                | MILESTONE (L20) | COMPARE (LC)
                | ROLLBACK (L21) | BISECT (L27) | AUDIT (L4) -> IDLE

Each tick, in order (6.2):
  1. lab work, no ssh: the last snapshot and the lab's files are folded into
     the evidence; the diagnoses (diagnose_stream D20-D33 and the lanes' own
     items) run on it; the stop-losses and holds apply; each idle lane takes the
     first item its diagnoses call for; lab items (L15, L26, L16L, L16RL, L16S)
     start as detached processes, and the ones running are polled;
  2. the one ssh of the tick: the ready items of every lane go to the cluster
     in one executor.submit_many batch, in priority order (TRAIN, DATA, MAINT;
     an R3 item goes alone, since its envelope grant runs it in its own call);
     with none ready, one stream snapshot (executor.stream_snapshot) observes
     every lane;
  3. the health checks on that snapshot (D5-D7 and D14 on every live
     experiment, D10S, D26, D27) and the stream ledger's hash chain.

Never COMPLETE by itself (P7, 6.7): exhaustion moves the DATA lane to
WAIT_DATA with discovery backing off 7 -> 14 -> 30 days and a card; a person
may mark the campaign complete when every condition of 6.7 holds. Stop-losses
pause with a card: 2 consecutive failed steps hold a lane; 3 consecutive
zero-yield sources hold DATA; 2 held lanes, D7, D10S, D14, D27, a broken
stream ledger chain or 3 raising ticks pause the campaign.

Governance (6.5) is the executor's: R2 data levers run directly only with
data_autonomy on (a person's flag), a stream replay pass and the caps; R3
builds run within the envelope (autonomy 'envelope', granted by a person) or
wait for a person; R4 is a card. Module drift between the lab and the
cluster (S23) refuses every stream submission until they agree.

Domain-free (S13): the domain's facts come from its stream-domain config
(stream_domains/<domain>.json) and its funnel domain config; this module names
no class, source, exam or lab.

CLI (lab):
    python -m weed_optimizer_framework.tools.inc_autopilot.stream enable --name N --by human:<email>
        --domain D [--autonomy off|envelope] [--data-autonomy off|on] [--envelope-su SU]
        [--window-cap-su SU|none] [--daily-cap-su SU|none] [--alloc-reserve-su SU] [--collect-gb-envelope GB]
        [--collect-gb-daily GB|none] [--envelope-end-utc YYYY-MM-DDTHH:MM:SSZ] [--protocol-v3-accepted]
    python -m weed_optimizer_framework.tools.inc_autopilot.stream configure --name N --by human:<email>
        [the same settings as enable]
        (changes the settings only: it neither enables the campaign nor lifts a pause or a held lane,
        which `enable` does. `none` clears a time-based cap: daily_cap_su, window_cap_su and
        collect_gb_daily have no default since the 2026-10-04 amendment, docs/CONTINUOUS_LOOP.md 6.6,
        and a cleared one is no cap.)
    python -m weed_optimizer_framework.tools.inc_autopilot.stream status [--name N]
    python -m weed_optimizer_framework.tools.inc_autopilot.stream release --name N --by human:<email>
        (a person's release of every lane a stop-loss held: 2 consecutive failed steps, a step past
        stop_loss step_retries, the DATA lane's zero-yield run; the next tick frees each lane held
        before the release and clears its steps' failure counts. `enable` does the same.)
    python -m weed_optimizer_framework.tools.inc_autopilot.stream lab-run --spec FILE   (the detached runner)
"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

from . import budget as B
from . import diagnose as DG
from . import diagnose_stream as DS
from . import evidence as E
from . import executor as X
from . import levers_stream as LS
from . import model as M
from ..brain import approvals as AP

STATE_FORMAT = "inc-autopilot/stream-state/1"
LANES = ("DATA", "TRAIN", "MAINT")
# A quarantine (L24) is a stop, not a collection step: it has a lane of its
# own so a busy DATA lane never delays it, and it goes to the cluster first
# (6.2 step 2: stop/cancel, then TRAIN, then DATA, then MAINT).
ALL_LANES = ("STOP",) + LANES
SUBMIT_ORDER = ("STOP", "TRAIN", "DATA", "MAINT")
AUTO = M.AUTOPILOT_ACTOR
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
HUMAN_RE = re.compile(r"^human:[A-Za-z0-9][A-Za-z0-9@._+-]{0,126}$")
HEALTH_EXP = ("D5", "D6", "D7", "D14")
CARDS_KEEP = 80
REFUSALS_KEEP = 5
FAILED_IDS_KEEP = 200                # proposal ids of lane items that ended failed (never proposed again)
# The collection levers, whose attempts are bounded per source (7.5: 3 failed
# attempts close a source; a 4th attempt pauses, S15): the step bound
# (stop_loss step_retries) leaves them to that rule.
SOURCE_BOUND = ("L16", "L16L", "L16R", "L16RL", "L16S", "L16I", "L17")
# The fetch levers (a source's attempt), the ones that fetch on the lab (folded
# from the lab process, then synced: L16S), and a source review's two forms
# (L16R on the cluster, L16RL on the lab, chosen by the candidate's placement).
FETCH_LEVERS = ("L16", "L16L", "L16R", "L16RL")
LAB_FETCH_LEVERS = ("L16L", "L16RL")
REVIEW_LEVERS = ("L16R", "L16RL")
STALE_TO_PAUSE = 2
BUILD_LOST_SNAPSHOTS = 3
LOST_RUNS_MAX = 3            # a step killed from outside this many times in a row counts as failed
# An intake the offline resolver refused for class names (collect.intake:
# names_pending) is no failure: L26 resolves the names on the lab, L16S pushes
# the names layer, and the intake runs again. A source still pending after
# this many L26 rounds goes to a person (a card), never round again.
NAMES_ROUNDS_MAX = 1
# An L26 of a round that resolved only part of its names (an authority error,
# `collect names` status partial) runs again, this many times, before the
# round counts anyway.
NAMES_PARTIAL_MAX = 2
# The refusal line an intake refused for class names leaves in its job log
# (collect.NamesPending: "... run `collect names --source S` (lever L26) first").
NAMES_REFUSAL = "(lever L26)"
WAIT_REFUSALS = ("today's cap", "this month's window", "of the domain's", "the cluster is not reachable",
                 "Mongo's health", "the execution log", "no slurm_sh hook", "could not be locked",
                 "collides with another request", "could not be filed", "one ssh per tick")
# L-2 (docs/CONTINUOUS_LOOP.md 2.6): the stream campaign's defaults. The
# time-based caps have none since 2026-10-04 (decided by the owner, 6.6
# amendment): L-2's 350 SU monthly window, its 120 SU daily cap and the 50 GB
# daily byte cap only delayed healthy work (on 2026-10-04 the cluster sat idle
# about 9 h while a segment waited for the UTC day to turn). None means no cap
# (budget.fits, executor.stream_limits); a campaign may still declare one. The
# lifetime fuses keep their defaults: envelope_su, its end date and
# collect_gb_envelope.
STREAM_DEFAULTS = {"enabled": False, "paused_reason": None, "mode": "stream", "domain": None,
                   "protocol_package": None, "stream": {}, "goal": {"kind": "continuous"},
                   "autonomy": "off", "autonomy_granted_by": None, "data_autonomy": "off",
                   "envelope_su": 1000.0, "envelope_end_utc": "2026-12-31T23:59:59Z", "window": "month",
                   "window_cap_su": None, "daily_cap_su": None, "alloc_reserve_su": None,
                   "collect_gb_envelope": 200.0, "collect_gb_daily": None, "protocol_v3_accepted_by": None,
                   "reopened_sources": {},
                   "brain": {"enabled": False, "model": None}}
# What a person passes to clear a time-based cap (configure_stream, the CLI's
# `--daily-cap-su none`): the setting becomes None, which is no cap. Only these
# three may be cleared; the lifetime envelope and collect_gb_envelope are fuses
# and always hold a number.
CLEAR = "none"
CLEARABLE = ("window_cap_su", "daily_cap_su", "collect_gb_daily")
LANE_OF = {"L15": "DATA", "L26": "DATA", "LP": "DATA", "L16": "DATA", "L16L": "DATA", "L16R": "DATA",
           "L16RL": "DATA", "L16I": "DATA", "L16S": "DATA", "L17": "DATA", "L24": "STOP", "LH": "DATA",
           "L18": "TRAIN", "L19": "TRAIN", "L22": "TRAIN",
           "L20": "MAINT", "L21": "MAINT", "L23": "MAINT", "L23B": "MAINT", "L23N": "MAINT", "L23V": "MAINT",
           "L23E": "MAINT", "L23C": "MAINT", "L23D": "MAINT", "L23Z": "MAINT", "L23F": "MAINT", "L23G": "MAINT",
           "L23H": "MAINT", "L23I": "MAINT", "L23J": "MAINT", "L23K": "TRAIN", "L23L": "MAINT",
           "L30": "MAINT", "L31": "MAINT", "L32": "MAINT", "L32X": "TRAIN", "L33": "MAINT", "L34": "MAINT",
           "L25": "MAINT", "L27": "MAINT", "L28": "MAINT", "L4": "MAINT", "LV": "MAINT", "LI": "MAINT", "LA": "MAINT",
           "LC": "MAINT"}
# Record-only levers (a measurement arm's native-resolution rescore, L23N;
# E1's base v3 build, L23V, and its agnostic rescore, L23E, 2026-10-03; E2's
# rescore and verdict, L23C, 2026-10-04; E2-C's attribution rescore and
# record, L23D, 2026-10-04, later; the model-zoo audit, L23Z, 2026-10-04;
# E3's scores of one arm, L23F, and its verdict, L23G, 2026-10-05): a
# failure is a person's card, never a failed step of its lane (no stop-loss
# counts it), and the item is not proposed again (_failed); a submission whose
# outcome is unknown is followed by its job name and its record, never a pause
# (_follow_record_only). A qos refusal is a platform defect and holds the lane
# as for every lever (S21).
RECORD_ONLY_LEVERS = ("L23N", "L23V", "L23E", "L23C", "L23D", "L23Z", "L23F", "L23G", "L23H", "L23J", "L23K",
                      "L23L", "L30", "L31", "L33", "L32X", "L34")
# The failure card title of each record-only lever (_failed; "%s" is its --exp, or L23F's --arm)
RECORD_ONLY_TITLES = {"L23N": "Native-resolution rescore of %s failed (L23N)",
                      "L23V": "Base v3 build (splits v3, E1) failed (L23V)",
                      "L23E": "E1's agnostic rescore of %s failed (L23E)",
                      "L23C": "E2's 12-class rescore and verdict failed (L23C)",
                      "L23D": "E2-C's attribution rescore and record failed (L23D)",
                      "L23Z": "The model-zoo audit v1 (every detector under one protocol) failed (L23Z)",
                      "L23F": "E3's two-stage scores of arm %s failed (L23F)",
                      "L23G": "E3's verdict failed (L23G)",
                      "L23H": "V3F's data build failed (L23H)", "L23I": "V3F's build of %s failed (L23I)",
                      "L23J": "V3F's scores of arm %s failed (L23J)", "L23K": "V3F's RF-DETR %s job failed (L23K)",
                      "L23L": "V3F's verdict failed (L23L)",
                      "L30": "DS1's data step %s failed (L30)", "L31": "DS1's sheets %s failed (L31)",
                      "L32": "DS1's build of %s failed (L32)", "L32X": "DS1's long-path %s job failed (L32X)",
                      "L33": "DS1's scores of arm %s failed (L33)", "L34": "DS1's verdict %s failed (L34)"}
# V3F (2026-10-11, the v3 experiment family; docs/CONTINUOUS_LOOP.md, V3F.6.2-V3F.6.4): its five levers, all record
# only. A failure is one card and the item is proposed once more under a new id; after its second failure
# (v3fam.max_failures) it is recorded failed and never holds the family's other items (_v3fam_failed). An L23K
# seed job that ends TIMEOUT (killed at its 36 h walltime) fails D6 at once, never retried (V3F.7, L23K's
# falsifier); a retry of L23K's train names only the seeds not yet trained. Every item is priced and counted against
# the family cap (260 GPU-h, decision 25: running items at their estimates, ended ones settled from sacct or, for a
# build, from its experiment's report, and every earlier attempt of an item proposed again kept in its prior_gpuh;
# the person steps from the derived status): an item whose estimate would pass it is recorded "not run (family
# cap)" with one card and never retried, and the verdict's argv names it (--cap-refused) so the cluster's records
# say so too. An L23K item whose jobs are still queued V3FAM_DETR_OVERDUE_H after its submission is one card (the
# seeds' walltime is 36 h).
V3FAM_LEVERS = ("L23H", "L23I", "L23J", "L23K", "L23L")
# DS1 (2026-10-29; docs/CONTINUOUS_LOOP.md, Amendment 2026-10-29, DS.9.2-DS.9.3): its six levers, all record only. A
# failure is one card and the item is proposed once more under a new id; after its second failure (ds1.max_failures)
# it is recorded failed and never holds DS1's other items (_ds1_failed). Every item is priced and counted against
# DS1's family cap (260 GPU-h) and, a phase-1 item until verdict v2 or the date substitution, the phase reserve's line
# (174.2 GPU-h): an item past either is recorded "not run (family cap)" with one card and never retried. A data step
# (L30) may run again once it is done (its next proposal takes a new id); every other item runs once.
DS1_LEVERS = ("L30", "L31", "L32", "L32X", "L33", "L34")
DS1_NOT_RUN_CAP = "not run (family cap)"
DS1_LONG_OVERDUE_H = 37.0
V3FAM_NOT_RUN_CAP = "not run (family cap)"
# Deviation 5 (2026-10-24; docs/prereg/V3F_dev5_2026-10-24.md). On 2026-10-21 deviation 4's CLI-only change of
# inc2/v3fam.py left the family's reproduction records taken by other code, inc2.v3fam.require_equivalence refused
# every score job of six arms ("the code changed since dev_identity.json ran"), and the ticker counted each refusal as
# the arm's own failure: six arms recorded failed twice and passed to the verdict as --ticker-failed. Now a family
# job refused by such a precondition of the platform's own making (V3FAM_PLATFORM_REFUSAL_RE, read from the job's
# provenance: the reproduction records, the E3-frozen files, a superseded record's move) is never the arm's failure
# (_v3fam_platform_refusal): its fails are not counted, it is never recorded failed or named in --ticker-failed, its
# GPU-h stay counted (prior_gpuh), its id is retired, it is held (DS.V3FAM_HELD_PLATFORM: neither due nor ended, so
# the verdict waits) with one card naming the remedy, and released when the snapshot reports the records current
# again; after V3FAM_PLATFORM_MAX such refusals of one item it stays held for a person, never looped. While the
# snapshot reports a record taken by other code than the deployed one, no family score is proposed
# (diagnose_stream.v3fam_plan).
# Deviation 6 (2026-10-25; docs/prereg/V3F_dev6_2026-10-25.md): D1's passes on its two season exams run under the
# loader fix and its pre-flight (inc2.v3fam_loader); a refusal of that check ("deviation 6's loader check refuses": the
# exam's frames, the label cache's sizes, the letterbox, an identity record not taken under the fix) is about the
# platform's loader and records, never D1's detectors, so it is held as the platform's too. So is a refusal of E3's
# ground-truth and frame checks (V3FAM_E3_FRAME): the ground-truth boxes through the geometry against the exam's
# labels, the label rows of each box, an image's EXIF tag and its size after transposition; they read the exam's
# labels, its image headers and the loader's batch geometry, never the arm's weights, so they come out the same for
# every arm and seed. Both kinds are deterministic (the same code on the same exam refuses the same way), so such a
# refusal is held for a person at once (V3FAM_PERSON_REFUSAL_RE; never released by the records' state): a person finds
# why, and a recorded deviation's reopen block proposes the item again.
V3FAM_E3_FRAME = (r"E3's path refuses: [^\n]{1,300}?(?:: ground-truth box \d+ comes back through the geometry "
                  r"|: ground-truth box \d+ \(class \d+\) has no label box of its class"
                  r"|: \d+ ground-truth boxes in the batch, \d+ distinct label rows"
                  r"| has EXIF orientation | after EXIF transposition, Ultralytics read it as )")
V3FAM_LOADER_REFUSAL = "deviation 6's loader check refuses"
V3FAM_PLATFORM_REFUSAL_RE = re.compile(
    r"the code changed since [^\n]{1,120}? ran|is missing or did not pass: no family score is taken without it"
    r"|no longer hash as V3F\.1 records|the E-series records changed while this family job ran"
    r"|is never moved over another file|a superseded record could not be moved aside"
    r"|deviation 6's loader check refuses|" + V3FAM_E3_FRAME)
V3FAM_PERSON_REFUSAL_RE = re.compile(r"deviation 6's loader check refuses|" + V3FAM_E3_FRAME)
V3FAM_PLATFORM_MAX = 3
# the family jobs' provenance names (run_inc2_build.sh; L23J's v3fam_score_<arm>, L23I's its experiment)
V3FAM_PROV = {"L23H": "v3fam_data", "L23L": "v3fam_v1"}
V3FAM_DETR_OVERDUE_H = 37.0
V3FAM_CAP_EPS = 1e-6
# The zoo audit's rerun commands (a person's, on the login node; written records are kept, so both resume)
ZOO_RERUN = ("bash run_inc2_zoo.sh submit --version %(v)s --shards-a %(a)s --shards-c %(c)s --concurrency %(p)s "
             "--max-gpu-hours %(h)s (resumable: written records are kept; refused while a zoo job is queued); or, "
             "for failed shards only, sbatch -p GPU-shared --job-name=inc_zoo_%(v)s_score_a --array=<ids> "
             "run_inc2_zoo.sh score --version %(v)s --stage a|c, then sbatch -p GPU-shared "
             "--job-name=inc_zoo_%(v)s_report --dependency=afterany:<that job> run_inc2_zoo.sh report --version %(v)s")
# A measurement arm's build (L23B of a baselines item marked measure) whose job
# ran and ended without its experiment: record only as well (a card; no lane
# failure; DR0 skips the item, /stage/baselines says failed), since no lane
# waits for it and the next measure item would otherwise meet the same refusal
# and hold MAINT (2026-10-04, E2's six builds). Only a build that ran: a
# submission that failed or was refused before anything was queued (an sbatch
# socket timeout, squeue unavailable, a duplicate job name) is the lane's
# ordinary failure and is proposed again, as on every lever. A build that ended
# with no refusal line in its provenance (the job was killed from outside:
# NODE_FAIL, PREEMPTED, BOOT_FAIL, TIMEOUT, or no sacct state at all) is built
# again under a new id, up to LOST_RUNS_MAX times in a row, before it is
# record only; a refusal of the build itself (inc2.baseline's ERROR line) or a
# cancelled job is record only at once (_failed, `ended`).
MEASURE_BUILD_TITLE = "Measurement arm %s: its build ended without the experiment (L23B)"
# sacct states a person or the scheduler's operator chose (never retried by the stream)
CANCELLED_STATES = ("CANCELLED",)
# A callable taking the StreamRun and returning its lab runner (tests).
LAB_RUNNER = None
# Refusals the identical request meets again (retry: false): a Step 1 batch
# refused on its pins (the frozen verifier, the v2 LOCK), or a cutter that
# cannot fill; a person decides (X11, D22).
RETRY_FALSE_RE = re.compile(r"verifier|LOCK|pinned|pins? (mismatch|differ)", re.I)
# R3 items only a person decides that must not hold their lane while they
# wait: filed "parked" and adopted into the lane once approved (a funnel_F9
# release at its deadline would otherwise stop collection until a person acts).
PERSON_LEVERS = ("LH",)
# Runs the lanes lost (2026-10-03 07:44Z: the dashboard restarted after an L16S
# was executed and before the tick wrote its state, so the state came back
# without the running item): the run ids adopted back into a lane (never twice),
# and the proposal ids of the items that left a lane (_clear), which together
# with the lanes, done_keys, failed_ids and declined tell a lost run from one
# the lanes already followed (_adopt_lost_runs).
ADOPTED_KEEP = 200
ENDED_KEEP = 200
# The proposal ids of lane items that ended done ({id: {utc, lever, exp}}, state done_ids; older ones are read from
# the campaign ledger's item_done lines). A done id that a diagnosis calls for again is never run again (the executor
# runs an id once), never declined and never a stall: one done_recalled event and one card name the diagnosis that
# does not read its lever's result (2026-10-26: D31 called for the label audit L4 of 2026-10-05 on every tick, its id
# was filed as declined after the executor refused it as already run, and lane MAINT was carded stalled 6 times).
DONE_IDS_KEEP = 400
# The collector's refusals that are the source's own outcome, not the lane's failure (2026-10-07: two identical lab
# fetches of gh_geezacoleman__palmer_detection refused over_byte_plan held the DATA lane on its stop-loss for 19
# days). On a fetch or an intake of the source (SOURCE_REFUSAL_LEVERS) that exits 2 with one of these codes:
#   close   the collector closed the source (its action close; for a cluster job, whose log carries no action, the
#           codes the collector always closes on): the source is closed with the code and detail as its reason;
#   replan  over_byte_plan at the plan: proposed once more at a bounded larger plan (REPLAN_*; a new id), then
#           closed with its reason;
#   person  a decision the collector holds for a person (R3): the source held for a person's review (one card);
#   retry   the source's own failure (no data file in its listing): counted against the source (closed after its 3
#           attempts), never the lane's, and the source ranked after the others (D20).
# A close, a person's hold and a source closed after its retries are in the zero-yield run (S20: 3 sources in a row
# hold the lane), and REFUSAL_RUN_HOLD sources in a row refused with the same code hold the lane: a refusal that
# meets every source is the machine's, not the sources'. A refusal whose detail names the machine (ENVIRONMENT_RE: a
# missing module, a full or unwritable disk) and every other refusal (a transport, listing, describe or download
# error, missing credentials, the byte envelope, an unknown source, a crash, a killed run) stays the lane's ordinary
# failure, where a repeat on one source counts against that source only (_failed: the lane's stop-loss counts
# distinct sources).
SOURCE_REFUSAL_LEVERS = ("L16", "L16L", "L16R", "L16RL", "L16I")
# the DATA lane's steps of one source whose denial by a person closes the source (_person_denied)
DENIED_CLOSES = ("L16", "L16L", "L16I", "L16S", "L26")
SOURCE_REFUSAL_CLOSE = ("never_train", "quarantined", "superseded", "licence_refused", "copy_candidate",
                        "attempts_exhausted", "archive_escape", "registered_outside_intake", "closed")
SOURCE_REFUSAL_REPLAN = ("over_byte_plan",)
SOURCE_REFUSAL_PERSON = ("licence_unresolved", "image_level_only", "normalise_failed", "no_target_class",
                         "not_fetchable", "archive_unreadable", "copy_scan_pending", "over_source_cap")
SOURCE_REFUSAL_RETRY = ("no_data_files",)
ENVIRONMENT_RE = re.compile(r"ImportError|ModuleNotFoundError|No module named|not installed|Permission denied|"
                            r"No space left|Read-only file system|\[Errno|MemoryError")
REFUSAL_RUN_HOLD = 3
# The replanned fetch of a source refused over_byte_plan at its plan: the source's size when the provider stated it
# (Content-Length, a listed size) plus 10 %, else REPLAN_MULT times the refused plan, at least REPLAN_FLOOR_BYTES;
# never over what the per-source limit or the collect envelope leaves (a larger plan would be filed for a person and
# hold the DATA lane on one source; a killed attempt counts its whole plan)
REPLAN_MULT = 4
REPLAN_FLOOR_BYTES = 10 ** 9
# the collector's over_byte_plan details (collect.transport, collect.fetch): (pattern, the group of the source's
# stated size or None, the group of the byte cap the download met)
BYTE_CAP_FORMS = ((re.compile(r"(\d+) bytes listed, over the (\d+)-byte cap"), 1, 2),
                  (re.compile(r"Content-Length (\d+) over the (\d+)-byte cap"), 1, 2),
                  (re.compile(r"Content-Range total (\d+) over the (\d+)-byte cap"), 1, 2),
                  (re.compile(r"more than the (\d+)-byte cap"), None, 1),
                  (re.compile(r"more than (\d+) bytes"), None, 1))
BYTE_CAP_GB_RE = re.compile(r"fits in ([0-9]+\.[0-9]+) GB")
# the collector's refusal line in a job's log (collect.__main__: "[collect] <verb> refused: <code>: <detail>")
COLLECT_REFUSAL_RE = re.compile(r"\[collect\] [a-z]+ refused: ([a-z_]+): ([^\n]*)")
# An intake's failure that says the fetch's staging is gone or changed (collect.intake: read_fetch's not_fetched,
# check_blobs' StaleInput): the fetch is no longer on the cluster, so the source may be fetched anew (_intake_again)
STAGING_GONE_RE = re.compile(r"\bnot_fetched\b|StaleInput|no fetch record at|staging blob of .* (is missing|has \d+ "
                             r"bytes|changed in transit)|now holds another fetch")
# The label audit's frozen experiment, snapshotted again until its record carries the audit (_refreeze_pending)
AUDIT_ARTIFACT = "%s/audit/label_audit.json"
# A segment build the cutter refused for want of eligible images (inc2.stream
# build: "nothing cut for <exp>: short", or short_after_guard once the guard
# excluded rows): no failure while the queue holds fewer than M (_cut_short).
SHORT_RE = re.compile(r"nothing cut for \S+: (short|short_after_guard)\b")
# The campaign ledger's events that end a lane item and name its proposal
# (with lab_job_finished, which names the lab job): a run with one was followed
# to its end and is never taken back (_ledger_ended, _seed_ended, _run_followed)
ENDED_EVENTS = ("item_done", "failed", "withdrawn", "intake_names_pending")
# Login verbs done when they ran (_on_result; a lost run of one, _adopt_run)
DONE_AT_ONCE = ("inc_stream_commit", "inc_stream_rollback", "inc_stream_quarantine", "inc_stream_release",
                "inc_unblock_transient")


# ------------------------------------------------------------------ helpers
def ds1_audit_sources(value):
    """The sources of an owner's audit render as its proposal names them (a comma list, or a list), sorted."""
    if isinstance(value, str):
        value = value.split(",")
    return sorted(str(x).strip() for x in value or () if str(x).strip())


def step_key(lever, policy_params):
    """The key of one step of a lane (the attempts, the step's failed runs):
    the lever and its policy params (levers_stream.policy_params) without the
    price (est_gpu_hours), which may change between two proposals of the same
    step. The proposal side and the failure side compute it the same way, so a
    failed step is proposed again with the next attempt, under a new id."""
    return "%s:%s" % (lever, _sha({k: v for k, v in (policy_params or {}).items() if k != "est_gpu_hours"})[:12])


def _utc(t):
    return datetime.datetime.fromtimestamp(float(t), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _secs(stamp):
    try:
        return datetime.datetime.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
                          .encode("utf-8")).hexdigest()


def _fmt4(x):
    return "-" if not isinstance(x, (int, float)) or isinstance(x, bool) else "%.4f" % x


def _short(text, n=300):
    text = str(text)
    return text if len(text) <= n else text[:n] + "..."


def _refusal_text(text, n=900):
    """A family job's provenance refusal, shortened for the ledger and cards without losing what classes it: the
    log tail run_inc2_build.sh keeps (its last five lines) ends with the exception's own message, so the tail is
    kept; a refusal of the platform's own making (V3FAM_PLATFORM_REFUSAL_RE) found anywhere in the full text stays
    in the result (deviation 5: a head cut at 600 characters dropped the message of a 688-character traceback)."""
    text = str(text or "")
    if len(text) <= n:
        return text
    out = text[:120] + " ... " + text[-(n - 125):]
    m = V3FAM_PLATFORM_REFUSAL_RE.search(text)
    if m and not V3FAM_PLATFORM_REFUSAL_RE.search(out):
        out = text[:120] + " ... " + m.group(0) + " ... " + text[-(n - 140 - len(m.group(0))):]
    return out


def _record_only(d):
    """A measurement arm's D5 without OP_PAUSE: the arm is record only and no
    lane waits for it, so a block that is not transient is a person's card,
    never a pause, and never the firing stop-loss that refuses every envelope
    grant (executor._trigger_check). L7 for its transient units stays."""
    if "OP_PAUSE" not in (d.get("levers") or []):
        return d
    return dict(d, levers=[x for x in d["levers"] if x != "OP_PAUSE"], severity="warn",
                summary="%s (a measurement arm, record only: a card, not a pause)" % d.get("summary"),
                detail=dict(d.get("detail") or {}, record_only=True))


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _read_json(path):
    try:
        with open(str(path), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fetch_facts(rows):
    """{"sources": {source: [[epoch, bytes]]}, "open": {source: [epoch]}} of a
    machine's source ledger (stream_remote.fetch_facts on each source row:
    its 'fetched' events, and its fetch_started events no closing event
    follows), which the byte limits count (executor.fetched_bytes); None
    when a row carries none (a fold written before they were kept, or of a
    ledger not read whole) or one is unreadable."""
    out = {"sources": {}, "open": {}}
    for src, row in rows.items():
        ev = row.get("fetched_events") if isinstance(row, dict) else None
        op = row.get("open_fetches") if isinstance(row, dict) else None
        if not isinstance(ev, list) or not isinstance(op, list):
            return None
        got = []
        for e in ev:
            if not isinstance(e, (list, tuple)) or len(e) != 2 or _num(e[1]) is None:
                return None
            got.append([_secs(e[0]), _num(e[1])])
        out["sources"][str(src)] = got
        if op:
            out["open"][str(src)] = [_secs(t) for t in op]
    return out


# ------------------------------------------------------------------ paths
class StreamPaths(object):
    """The lab files of one domain's stream campaigns (model.domain_campaign_dir)."""

    def __init__(self, lab_repo=None, domain=None):
        self.domain = domain or M.DOMAIN
        self.lab_repo = lab_repo
        self.campaign_dir = M.domain_campaign_dir(self.domain, lab_repo)
        self.ledger = self.campaign_dir / "inc_campaign.jsonl"
        self.replay = self.campaign_dir / "replay"
        self.campaigns = self.campaign_dir / "campaigns"
        root = Path(lab_repo) if lab_repo is not None else M.LAB_REPO
        self.lab_inc = root / "results" / "framework" / "inc"

    def state(self, name):
        return self.campaigns / name / "state.json"

    def latest(self, name):
        return self.campaigns / name / "latest_stream_snapshot.json"

    def diagnoses(self, name):
        return self.campaigns / name / "diagnoses.json"

    def frozen(self, name, exp):
        return self.campaigns / name / "frozen" / ("%s.json" % exp)

    def lab_jobs(self, name):
        return self.campaigns / name / "lab_jobs"

    def candidates(self):
        return self.lab_inc / "collect" / "candidates.json"

    def staging(self):
        return self.lab_inc / "intake" / "staging"

    def names_dir(self):
        return self.lab_inc / "intake" / "names"

    def cand_rel(self, source):
        """INC_DIR-relative path of one discovered source's candidate record, as
        synced to the cluster for its fetch there (collect fetch --candidates)."""
        return "intake/candidates_sync/%s.json" % re.sub(r"[^A-Za-z0-9_.-]", "_", str(source))[:160]


# ------------------------------------------------------------------ config
def stream_config(raw, name):
    c = copy.deepcopy(STREAM_DEFAULTS)
    for k, v in (raw if isinstance(raw, dict) else {}).items():
        c[k] = copy.deepcopy(v)
    c["stream"] = dict(c.get("stream") or {})
    c["name"] = name
    return c


def check_stream_config(cfg):
    """[problems] of a stream campaign's config; [] when the ticker can run it."""
    out = []
    if cfg.get("mode") != "stream":
        out.append("mode is %r, not 'stream'" % cfg.get("mode"))
    if not isinstance(cfg.get("domain"), str) or not re.match(r"^[a-z][a-z0-9_]{0,39}$", cfg["domain"]):
        out.append("a stream campaign names its domain (domain: %r)" % cfg.get("domain"))
    if (cfg.get("goal") or {}).get("kind") != "continuous":
        out.append("a stream campaign's goal is {'kind': 'continuous'}, not %r" % cfg.get("goal"))
    for k in ("autonomy",):
        if cfg.get(k) not in ("off", "envelope"):
            out.append("%s is 'off' or 'envelope', not %r" % (k, cfg.get(k)))
    if cfg.get("data_autonomy") not in ("off", "on"):
        out.append("data_autonomy is 'off' or 'on', not %r" % cfg.get("data_autonomy"))
    return out


def configure_stream(name, by, domain=None, enable=None, autonomy=None, data_autonomy=None, envelope_su=None,
                     window_cap_su=None, daily_cap_su=None, alloc_reserve_su=None, collect_gb_envelope=None,
                     protocol_v3_accepted=None, stream_domain=None, collect_config=None, cfg_hooks=None,
                     lab_repo=None, clock=None, complete=None, envelope_end_utc=None, collect_gb_daily=None,
                     reopen_source=None, reopen_why=None):
    """Create or change a stream campaign as person `by`. data_autonomy,
    autonomy 'envelope', the acceptance of Protocol v3 and a completion are a
    person's flags (6.5, 10): the ticker never writes them. A setting left
    None is unchanged; CLEAR ("none") clears one of CLEARABLE (a time-based
    cap) to None, which is no cap."""
    from . import campaign as C
    if not NAME_RE.match(str(name or "")):
        raise ValueError("campaign name %r is not valid" % (name,))
    if not HUMAN_RE.match(str(by or "")):
        raise ValueError("a stream campaign is configured by a person (human:<email>), not %r" % (by,))
    for label, v in (("envelope_su", envelope_su), ("window_cap_su", window_cap_su), ("daily_cap_su", daily_cap_su),
                     ("alloc_reserve_su", alloc_reserve_su), ("collect_gb_envelope", collect_gb_envelope),
                     ("collect_gb_daily", collect_gb_daily)):
        if v == CLEAR and label in CLEARABLE:
            continue
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0):
            raise ValueError("%s must be a number >= 0%s, got %r"
                             % (label, " or %r" % CLEAR if label in CLEARABLE else "", v))
    if envelope_end_utc is not None and _secs(envelope_end_utc) is None:
        # the envelope's end pauses the campaign (envelope_ended): a person
        # sets the new end here, never by hand-editing the config
        raise ValueError("envelope_end_utc must read YYYY-MM-DDTHH:MM:SSZ, got %r" % (envelope_end_utc,))
    cfg_hooks = cfg_hooks or C.default_cfg_hooks()
    now = (clock or time.time)()
    utc = _utc(now)

    def change(c):
        c.setdefault("mode", "stream")
        if c.get("mode") != "stream":
            raise ValueError("campaign %s is an experiment-mode campaign; a stream campaign is a new one" % name)
        c.setdefault("goal", {"kind": "continuous"})
        if domain is not None:
            c["domain"] = domain
        dom = LS.load_domain(stream_domain or c.get("stream", {}).get("stream_domain") or c.get("domain"))
        st = dict(c.get("stream") or {})
        st.setdefault("sid", dom["sid"])
        st.setdefault("stream_domain", stream_domain or dom["domain"])
        st.setdefault("collect_config", collect_config or dom.get("collect_config"))
        st.setdefault("K_max", (dom.get("increment") or {}).get("K_max"))
        c["stream"] = st
        c.setdefault("protocol_package", dom.get("protocol_package"))
        if autonomy is not None:
            if autonomy not in ("off", "envelope"):
                raise ValueError("autonomy is 'off' or 'envelope'")
            c["autonomy"] = autonomy
            if autonomy == "envelope":
                c["autonomy_granted_by"] = by
        if data_autonomy is not None:
            if data_autonomy not in ("off", "on"):
                raise ValueError("data_autonomy is 'off' or 'on'")
            c["data_autonomy"] = data_autonomy
            c["data_autonomy_by"] = by
            c["data_autonomy_utc"] = utc
        for k, v in (("envelope_su", envelope_su), ("window_cap_su", window_cap_su), ("daily_cap_su", daily_cap_su),
                     ("alloc_reserve_su", alloc_reserve_su), ("collect_gb_envelope", collect_gb_envelope),
                     ("collect_gb_daily", collect_gb_daily)):
            if v == CLEAR:
                c[k] = None
            elif v is not None:
                c[k] = float(v)
        if envelope_end_utc is not None:
            c["envelope_end_utc"] = str(envelope_end_utc)
        if protocol_v3_accepted:
            c["protocol_v3_accepted_by"] = by
        if complete:
            c["completed_by"], c["completed_utc"] = by, utc
        if reopen_source is not None:
            if not re.match(r"^[A-Za-z0-9_.-]{1,200}$", str(reopen_source)):
                raise ValueError("source %r is not a source id" % (reopen_source,))
            if not str(reopen_why or "").strip():
                raise ValueError("reopening a closed source records why (--why)")
            ro = dict(c.get("reopened_sources") or {})
            ro[str(reopen_source)] = {"by": by, "utc": utc, "why": str(reopen_why)[:500]}
            c["reopened_sources"] = ro
        if enable is not None:
            c["enabled"] = bool(enable)
            if enable:
                c["paused_reason"] = None
                c["resumed_utc"] = utc
        c.setdefault("enabled", False)
        c["updated_by"], c["updated_utc"] = by, utc
        return c
    new = C._update_config(cfg_hooks, name, change)
    full = stream_config(new, name)
    bad = check_stream_config(full)
    if bad:
        raise ValueError("; ".join(bad))
    paths = StreamPaths(C.lab_repo_arg(lab_repo) if lab_repo else None, full["domain"])
    _append(paths.ledger, {"utc": utc, "campaign": name, "event": "configured", "decided_by": by,
                           "config": {k: full.get(k) for k in ("enabled", "domain", "autonomy", "data_autonomy",
                                                               "envelope_su", "envelope_end_utc", "window_cap_su",
                                                               "daily_cap_su", "alloc_reserve_su", "collect_gb_envelope",
                                                               "collect_gb_daily", "protocol_v3_accepted_by",
                                                               "reopened_sources",
                                                               "stream")}})
    return full


def _append(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, sort_keys=True, default=str) + "\n")


def blank_state(name, cfg):
    return {"format": STATE_FORMAT, "name": name, "ticks": 0, "errors": 0, "paused": None,
            "lanes": {ln: {"phase": "IDLE", "item": None, "fails": 0, "hold": None, "diag_hold": None,
                           "until_utc": None} for ln in ALL_LANES},
            "sources": {}, "zero_run": [], "segments": [], "milestones": [], "exps": [], "frozen": {},
            "stage": {"r0": {}}, "discover": {}, "cards": [], "card": None, "notes": {},
            "refusals": [], "declined": [], "source_reviews": {}, "person_items": {}, "history": {}, "stale": {},
            "d33_history": [],
            "attempts": {}, "rollbacks": [], "failed_ids": [], "step_failures": {},
            "doublings": 0, "drift": None, "decisions_logged": False, "prospective": None, "bisected": None,
            "last_snapshot_utc": None, "last_tick_utc": None, "snapshot_failures": 0, "updated_utc": None}


def load_state(paths, name):
    st = _read_json(paths.state(name))
    return st if isinstance(st, dict) and st.get("format") == STATE_FORMAT else None


# ------------------------------------------------------------------ the lab runner
LAB_TIMEOUT_S = 6 * 3600                  # a lab job's wall clock unless it moves bytes (below)
LAB_FETCH_MIN_RATE = 1.0e6                # bytes/s a lab fetch is still worth waiting for
LAB_FETCH_TIMEOUT_MAX_S = 48 * 3600


def lab_job_timeout(kind, max_bytes):
    """The lab runner's wall clock for one job. A fetch is sized from the bytes
    it may move: 2 h plus max_bytes at 1 MB/s, between 6 h and 48 h. The flat
    6 h cut a 49.7 GB single-file fetch (zenodo_15808623, about 5.8 h at the
    2.4 MB/s the lab measured on 2026-10-02) at the limit, and a killed fetch
    restarts from zero: the collector resumes only within its own process.
    The collector's own deadline (base_timeout_s + size / min_rate_bytes_per_s)
    still decides a stalled download well before this."""
    if kind != "fetch":
        return LAB_TIMEOUT_S
    try:
        b = float(max_bytes)
    except (TypeError, ValueError):
        return LAB_TIMEOUT_S
    if b <= 0:
        return LAB_TIMEOUT_S
    return int(min(LAB_FETCH_TIMEOUT_MAX_S, max(LAB_TIMEOUT_S, 2 * 3600 + b / LAB_FETCH_MIN_RATE)))


class LabRunner(object):
    """Detached lab processes (S22: a lab fetch never blocks the tick). launch()
    starts `python -m ...stream lab-run --spec FILE` in its own session and
    returns at once; the child runs the argv and writes FILE's result path."""

    def __init__(self, root, cwd=None, python=None):
        self.root, self.cwd, self.python = Path(root), cwd or str(LS.CODE_ROOT), python or sys.executable

    def launch(self, job, argv, timeout=LAB_TIMEOUT_S):
        self.root.mkdir(parents=True, exist_ok=True)
        spec = {"job": job, "argv": [str(a) for a in argv], "result": str(self.root / ("%s.result.json" % job)),
                "timeout": int(timeout), "cwd": self.cwd}
        sp = self.root / ("%s.spec.json" % job)
        _write_json(sp, spec)
        env = dict(os.environ)
        env["PYTHONPATH"] = self.cwd + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        proc = subprocess.Popen([self.python, "-m", "weed_optimizer_framework.tools.inc_autopilot.stream", "lab-run",
                                 "--spec", str(sp)], cwd=self.cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        _write_json(sp, dict(spec, pid=proc.pid))
        return {"job": job, "spec": str(sp)}

    def poll(self, job):
        """The job's result, None while it runs, or a failure when its
        process is gone without writing one. A lab-run killed from outside (a
        dashboard restart under KillMode=control-group, an OOM kill, a reboot)
        never writes its result, and the lane that follows it waited for ever:
        on 2026-10-03 a deploy killed the zenodo_15808623 fetch and its L16L
        item stayed 'running' with nothing behind it."""
        rp = self.root / ("%s.result.json" % job)
        res = _read_json(rp)
        if res is not None:
            return res
        sp = self.root / ("%s.spec.json" % job)
        spec = _read_json(sp)
        if not isinstance(spec, dict):
            return None
        if _lab_run_alive(spec, sp):
            return None
        res = _read_json(rp)            # it may have finished between the two reads
        if res is not None:
            return res
        return {"ok": False, "rc": None, "lost": True, "job": job, "finished_utc": M.utc_now(),
                "error": "the lab process of %s is gone and wrote no result (killed from outside: a restart, "
                         "an OOM kill or a reboot)" % job}


def _lab_run_alive(spec, spec_path):
    """Whether a lab-run process of this spec is still running: its recorded
    pid when it still runs 'lab-run --spec <spec_path>', else (a spec written
    before pids were recorded, or a reused pid) any process whose command
    line names the spec. Unknown (no process table to read) counts as alive."""
    marker = str(spec_path)
    pid = spec.get("pid")
    cmd = _cmdline(pid) if isinstance(pid, int) and pid > 0 else None
    if cmd is not None and marker in cmd:
        return True
    try:
        out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return True
    if not out.strip():
        return True
    for line in out.splitlines():
        if marker in line and "lab-run" in line:
            return True
    return False


def _cmdline(pid):
    """The command line of a running pid ('' when gone), or None when it
    cannot be read."""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return ""
    except (PermissionError, OSError, ValueError):
        pass
    try:
        raw = Path("/proc/%d/cmdline" % int(pid)).read_bytes()
        return raw.replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        try:
            return subprocess.run(["ps", "-o", "args=", "-p", str(int(pid))], capture_output=True, text=True,
                                  timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            return None


def lab_run(spec_path):
    """The detached child: run the spec's argv, write its result atomically."""
    spec = _read_json(spec_path) or {}
    t0 = time.time()
    try:
        p = subprocess.run(spec["argv"], cwd=spec.get("cwd"), capture_output=True, text=True,
                           timeout=int(spec.get("timeout") or 3600))
        res = {"ok": p.returncode == 0, "rc": p.returncode, "tail": (p.stdout or "")[-2000:],
               "stderr_tail": (p.stderr or "")[-800:]}
    except (OSError, subprocess.TimeoutExpired, KeyError) as e:
        res = {"ok": False, "rc": None, "error": "%s: %s" % (type(e).__name__, e)}
    res.update(job=spec.get("job"), seconds=round(time.time() - t0, 3), finished_utc=M.utc_now())
    _write_json(spec.get("result") or (str(spec_path) + ".result.json"), res)
    return 0 if res.get("ok") else 1


def sync_argv(lab_inc, source, target, data_target, file=None, names=False):
    """The lab -> cluster push of one source's staging (L16S), or of one
    discovered source's candidate record (`file`, under intake/candidates_sync/),
    or of the collector's names layer for it (`names`), as the funnel's sync:
    rsync of the listed files, then a sha256 check of each on arrival."""
    return [sys.executable, "-m", "weed_optimizer_framework.tools.inc_autopilot.stream", "lab-sync",
            "--lab-inc", str(lab_inc), "--source", str(source), "--target", str(target or ""),
            "--data-target", str(data_target or "")] + (["--file", str(file)] if file else []) + \
        (["--names"] if names else [])


CAND_SYNC_DIR = "intake/candidates_sync"
# The collector's names layer (collect.names: INC_DIR/intake/names/, written by
# L26 on the lab only, since the authority needs the network). Every collector
# job on the cluster reads it offline, so a source whose names L26 resolved is
# refused there as names-pending until the layer is pushed.
NAMES_DIR = "intake/names"


def names_files(root, source):
    """The names-layer files of one source that exist under the lab INC_DIR
    `root`: the layer itself and the source's resolution record (the file name
    as collect.names writes it, collect.safe_name)."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(source)).strip("_") or "x"
    rels = ["%s/names_cache.json" % NAMES_DIR, "%s/names_%s.json" % (NAMES_DIR, safe)]
    return [r for r in rels if (Path(root) / r).is_file()]


def lab_sync(lab_inc, source, target, data_target=None, runner=None, cluster_inc=None, repo=None, file=None,
             names=False):
    """Push INC_DIR/intake/staging/<source>/ (or the one file `file`, under
    intake/candidates_sync/; or, with `names`, the names layer and the
    source's names record under intake/names/) to the cluster and verify every
    file on arrival (sha256), refusing any path outside it. A staging push
    carries the source's names files too, so the cluster's intake reads the
    names L26 resolved on the lab. {"ok", "pushed", "error"}."""
    run = runner or subprocess.run
    root = Path(lab_inc)
    if names:
        files = names_files(root, source)
        if not files:
            return {"ok": False, "pushed": [], "error": "no names layer under %s/%s" % (root, NAMES_DIR)}
    elif file:
        rel = Path(str(file))
        if rel.is_absolute() or ".." in rel.parts or str(rel.parent) != CAND_SYNC_DIR or not (root / rel).is_file():
            return {"ok": False, "pushed": [], "error": "%s is not a candidate record under %s" % (file, CAND_SYNC_DIR)}
        files = [str(rel)]
    else:
        base = root / "intake" / "staging" / str(source)
        if not base.is_dir():
            return {"ok": False, "pushed": [], "error": "no staging for %s" % source}
        files = []
        for d, _dirs, fs in os.walk(str(base)):
            for f in fs:
                files.append(str((Path(d) / f).relative_to(root)))
        files = sorted(files) + names_files(root, source)
    if any(".." in Path(f).parts for f in files):
        return {"ok": False, "pushed": [], "error": "a path outside the staging"}
    if not target:
        return {"ok": False, "pushed": [], "error": "no cluster ssh target (CLUSTER_SSH)"}
    shas = {rel: LS.sha256_file(root / rel) for rel in files}     # streamed: staging blobs run to GBs
    cluster_inc = cluster_inc or M.CLUSTER_INC_DIR
    p = run(["rsync", "-a", "--files-from=-", "--", str(root) + "/", "%s:%s/" % (data_target or target, cluster_inc)],
            input="\n".join(files) + "\n", capture_output=True, text=True, timeout=6 * 3600)
    if p.returncode != 0:
        return {"ok": False, "pushed": [], "error": "rsync exited %d: %s" % (p.returncode, (p.stderr or "")[-300:])}
    c = run(["ssh", target, X._remote_py(repo or M.CLUSTER_REPO, X._SYNC_CHECK_PY, [cluster_inc, "0"])],
            input=json.dumps(shas, sort_keys=True), capture_output=True, text=True, timeout=600)
    return {"ok": c.returncode == 0, "pushed": files, "sha256": shas,
            "error": "" if c.returncode == 0 else "the arrival check failed: %s" % ((c.stdout or "") + (c.stderr or ""))[-300:]}


def _byte_cap_of(detail):
    """(cap, size, exact) of a collector over_byte_plan detail (BYTE_CAP_FORMS, BYTE_CAP_GB_RE): the byte cap the
    refused download met, the source's size when the provider stated it (else None), and whether the cap is exact
    (False: 'no selected file fits in X GB', X to 0.01 GB). (None, None, False) for a detail of no known form."""
    for rx, gs, gc in BYTE_CAP_FORMS:
        m = rx.search(str(detail or ""))
        if m:
            return int(m.group(gc)), (int(m.group(gs)) if gs else None), True
    m = BYTE_CAP_GB_RE.search(str(detail or ""))
    if m:
        return int(round(float(m.group(1)) * 1e9)), None, False
    return None, None, False


# ------------------------------------------------------------------ the ticker
class StreamRun(object):
    """One stream campaign's step in one tick (campaign._tick_all dispatches here)."""

    def __init__(self, name, raw, paths, ssh, clock, log, cfg_hooks, resources, domain_budget, log_action,
                 preamble):
        self.name, self.ssh, self.clock, self.log = name, ssh, clock, log
        self.cfg_hooks, self.log_action, self.resources = cfg_hooks, log_action, resources
        self.cfg = stream_config(raw, name)
        self.now = clock()
        self.utc = _utc(self.now)
        self.domain = M.campaign_domain(self.cfg)
        self.paths = StreamPaths(paths.lab_repo, self.domain)
        self.st = None
        self.ev = None
        self.diags = []
        self.dom = None
        self.th = None
        self.prior = None
        self.domain_budget = domain_budget
        self.preamble = preamble
        self._execs = None                       # this campaign's execution records, read once a tick (_executions)
        self._ended = None                       # the items the campaign ledger records ended (_ledger_ended)
        self._ended_done = None                  # of those, the ones that ended done ({id: {utc, lever, exp}})
        self._ctx = self._payload = None
        self.xctx = X.Context(slurm_sh=ssh, resources=resources, domain_budget=domain_budget, clock=clock,
                              lab_repo=paths.lab_repo, preamble=preamble, domain=self.domain,
                              diagnoses=lambda _n: self._hook_diagnoses())
        self.runner = LAB_RUNNER(self) if LAB_RUNNER is not None else LabRunner(self.paths.lab_jobs(name))
        self.xctx.local_hooks.update(self._lab_hooks())

    # ---- plumbing
    @property
    def sid(self):
        return str((self.cfg.get("stream") or {}).get("sid") or (self.dom or {}).get("sid"))

    @property
    def pkg(self):
        return self.cfg.get("protocol_package") or (self.dom or {}).get("protocol_package")

    def _fired(self):
        return [d for d in self.diags if isinstance(d, dict) and d.get("fired")]

    def _hook_diagnoses(self):
        """The executor's diagnoses hook (as campaign._Run's): the fired
        diagnoses of this tick, and those each lane item was proposed from --
        a lane-gated diagnosis (D22, D24) goes silent once its item holds the
        lane, and the envelope still checks the item against the diagnosis
        that proposed it. Stop-loss and health diagnoses are this tick's."""
        now = {d.get("id"): d for d in self._fired()}
        out = list(now.values())
        for ln in ALL_LANES:
            it = ((self.st or {}).get("lanes") or {}).get(ln, {}).get("item") or {}
            for d in it.get("diagnoses") or []:
                if isinstance(d, dict) and d.get("fired") and d.get("id") not in now:
                    now[d["id"]] = d
                    out.append(d)
        return out

    def _ledger(self, event, **kw):
        rec = {"utc": self.utc, "campaign": self.name, "event": event, "mode": "stream",
               "tick": (self.st or {}).get("ticks"), "decided_by": kw.pop("decided_by", AUTO)}
        rec.update(kw)
        try:
            _append(self.paths.ledger, rec)
        except OSError as e:
            self.log.warning("[inc-stream] %s: ledger not written: %s" % (self.name, e))
        return rec

    def _once(self, key, value, event, **kw):
        notes = self.st.setdefault("notes", {})
        h = _sha(value)
        if notes.get(key) == h:
            return None
        notes[key] = h
        return self._ledger(event, **kw)

    def _card(self, kind, title, detail="", lever=None, trigger=None):
        st = self.st
        key = "%s:%s:%s" % (kind, lever or "", _sha([title, detail])[:12])
        if any(c.get("key") == key for c in st.get("cards") or []):
            st["card"] = {"kind": kind, "title": title, "detail": detail, "utc": self.utc}
            return
        rec = {"key": key, "kind": kind, "lever": lever, "title": title, "detail": _short(detail, 2000),
               "risk": "R4" if lever and str(lever).startswith("X") else None, "utc": self.utc,
               "trigger": list(trigger or [])}
        st["cards"] = (list(st.get("cards") or []) + [rec])[-CARDS_KEEP:]
        st["card"] = {"kind": kind, "title": title, "detail": _short(detail, 2000), "utc": self.utc}
        self._ledger("card", kind=kind, lever=lever, title=title, detail=_short(detail, 1000),
                     trigger=list(trigger or []))

    def _pause(self, reason, detail=""):
        from . import campaign as C
        st = self.st
        st["paused"] = {"reason": reason, "utc": self.utc, "by": AUTO, "config_written": False}
        self._card("paused", "Paused: " + _short(reason, 160), detail or reason)

        def change(c):
            c.update(enabled=False, paused_reason=reason)
            return c
        try:
            C._update_config(self.cfg_hooks, self.name, change)
            self.cfg.update(enabled=False, paused_reason=reason)
            st["paused"]["config_written"] = True
        except Exception as e:
            self.log.warning("[inc-stream] %s: pause not written to the config (%s)" % (self.name, e))
        self._ledger("paused", reason=reason)
        self.log.error("[inc-stream] %s PAUSED: %s" % (self.name, reason))

    def _paused_reason(self):
        if not self.cfg.get("enabled"):
            return self.cfg.get("paused_reason") or "not enabled"
        return self.cfg.get("paused_reason") or ((self.st.get("paused") or {}).get("reason"))

    def _reopen_sources(self):
        """A person's reopening of a closed source (`stream reopen`): the
        config stamps {source: {by, utc, why}}; a source closed before that
        stamp becomes a candidate again with its failure count reset, once
        per stamp, and the ledger records who and why. A quarantine is not
        a closure and is lifted only by inc2.stream unquarantine."""
        ro = self.cfg.get("reopened_sources") or {}
        srcs = self.st.setdefault("sources", {})
        for src, rec in sorted(ro.items()):
            rec = rec or {}
            s = srcs.get(src)
            if s and s.get("status") == "held" and s.get("names_held") and rec.get("utc") \
                    and not (s.get("reopened_utc") and str(s["reopened_utc"]) >= str(rec["utc"])):
                # held after its names rounds: a person mapped the names (a card table, the names layer), so
                # the intake runs again with its rounds reset
                prev = {k: s.get(k) for k in ("status", "held_reason", "names_rounds", "pending_names")}
                s.update(status="fetched", held_by=None, held_reason=None, names_held=None, names_rounds=0,
                         reopened_utc=rec["utc"], reopened_by=rec.get("by"))
                self._ledger("source_reopened", source=src, decided_by=rec.get("by") or "human",
                             reasons=[_short(rec.get("why") or "", 300)], before=prev)
                continue
            if s and s.get("status") == "held" and s.get("held_by") == "stream" and rec.get("utc") \
                    and not (s.get("reopened_utc") and str(s["reopened_utc"]) >= str(rec["utc"])) \
                    and not (s.get("held_utc") and str(rec["utc"]) < str(s["held_utc"])):
                # held by the stream for a person on a collector refusal (_source_refused): a person's review
                # releases it to be collected again
                prev = {k: s.get(k) for k in ("status", "held_by", "held_reason", "held_utc")}
                s.update(status="shard_pending" if DS.mid_shards(s) else "candidate", held_by=None,
                         held_reason=None, reopened_utc=rec["utc"], reopened_by=rec.get("by"))
                self._ledger("source_reopened", source=src, decided_by=rec.get("by") or "human",
                             reasons=[_short(rec.get("why") or "", 300)], before=prev)
                continue
            if not s or s.get("status") != "closed" or not rec.get("utc"):
                continue
            if s.get("reopened_utc") and str(s["reopened_utc"]) >= str(rec["utc"]):
                continue
            if s.get("closed_utc") and str(rec["utc"]) < str(s["closed_utc"]):
                continue                             # closed again after the stamp: a new stamp is needed
            prev = {k: s.get(k) for k in ("status", "failures", "attempts", "closed_reason", "closed_utc")}
            # attempts too: the 4th collection attempt on a source pauses the campaign (S15), so a reopened
            # source whose attempts stayed at 3 would stop the stream on its first retry. A source closed part-way
            # through its intake shards resumes them (a new fetch record would start the source over)
            s.update(status="shard_pending" if DS.mid_shards(s) else "candidate", failures=0, attempts=0,
                     closed_reason=None, reopened_utc=rec["utc"], reopened_by=rec.get("by"))
            if s["status"] == "shard_pending":
                self._bump_attempts("L16I", {"source": src})        # its next intake runs under an id of its own
            self._ledger("source_reopened", source=src, decided_by=rec.get("by") or "human",
                         reasons=[_short(rec.get("why") or "", 300)], before=prev)

    def _resumed(self):
        st = self.st
        p = st.get("paused") or {}
        r = self.cfg.get("resumed_utc")
        if not p and r and self.cfg.get("enabled") and not self.cfg.get("paused_reason"):
            # a lane held by a stop-loss while the campaign ran on: a person's
            # resume (`stream enable`) after the hold releases it; without this
            # a single held lane could only be released by a pause
            for ln in LANES:
                lane = st["lanes"][ln]
                if lane.get("hold") and lane.get("hold_utc") and str(r) >= str(lane["hold_utc"]):
                    self._ledger("lane_released", lane=ln, hold=lane["hold"],
                                 decided_by=self.cfg.get("updated_by") or "human")
                    lane["hold"], lane["fails"], lane["hold_utc"] = None, 0, None
                    lane["fail_sources"] = []
                    self._release_steps(ln)
                    if ln == "DATA":
                        st["zero_run"] = []
                        st["refusal_run"] = []
        if not p or not self.cfg.get("enabled") or self.cfg.get("paused_reason"):
            return
        if not ((r and str(r) >= str(p.get("utc") or "")) or p.get("config_written")):
            return
        st["paused"] = None
        st["errors"] = 0
        for ln in LANES:                             # a person's resume clears the lanes' stop-loss holds
            lane = st["lanes"][ln]
            if lane.get("hold"):
                self._ledger("lane_released", lane=ln, hold=lane["hold"])
                self._release_steps(ln)
            lane["hold"], lane["fails"] = None, 0
            lane["fail_sources"] = []
        st["zero_run"] = []
        st["refusal_run"] = []
        if (st.get("card") or {}).get("kind") == "paused":
            st["card"] = None
        self._ledger("resumed", decided_by=self.cfg.get("updated_by") or "human")

    @property
    def camp(self):
        """The campaign as the executor reads it (executor._campaign)."""
        c, st = self.cfg, self.st or {}
        lanes = st.get("lanes") or {}
        in_flight = {}
        for ln in ALL_LANES:
            it = (lanes.get(ln) or {}).get("item")
            if it and it.get("status") in ("executed", "running"):
                fam = LS.family(it["lever"]) if it.get("lever") in LANE_OF else it.get("lever")
                in_flight[fam] = in_flight.get(fam, 0) + 1
        q = self._artifact("stream/%s/queue_summary.json" % self.sid) if self.st else None
        pend = [x for x in ((q or {}).get("rollback_pending") or []) if isinstance(x, dict) and x.get("to_pool")]
        return {"name": self.name, "mode": "stream", "autonomy": c.get("autonomy") or "off",
                "autonomy_granted_by": c.get("autonomy_granted_by"), "envelope_su": c.get("envelope_su"),
                "daily_cap_su": c.get("daily_cap_su"), "window_cap_su": c.get("window_cap_su"),
                "data_autonomy": c.get("data_autonomy") or "off",
                "paused_reason": c.get("paused_reason") or (st.get("paused") or {}).get("reason"),
                "last_milestone_pool": pend[-1]["to_pool"] if pend else None,
                "in_flight": in_flight, "limit_counts": self._limit_counts(),
                "collect_gb_envelope": c.get("collect_gb_envelope"), "collect_gb_daily": c.get("collect_gb_daily"),
                "fetched": self._fetched()}

    def _limit_counts(self):
        st = self.st or {}
        q = (self._artifact("stream/%s/queue_summary.json" % self.sid) if self.st else None) or {}
        pend = [x for x in q.get("rollback_pending") or [] if isinstance(x, dict)]
        rb = st.get("rollbacks") or []
        out = {"L21": {"per_milestone": sum(1 for r in rb if pend and r.get("milestone") == pend[-1].get("milestone"))},
               "L27": {"per_rollback": 1 if rb and st.get("bisected") == rb[-1].get("utc") else 0},
               "L28": {"per_stream_version": 1 if ((st.get("stage") or {}).get("r0") or {}).get("stage_c_built")
                       else 0}}
        return out

    def _fetched(self):
        """What the executor's byte limits count fetched bytes from
        (executor.fetched_bytes, 6.6): the fetch attempts whose end the lane
        saw, the cluster collector's fetched facts (_fetch_facts) as of the
        last snapshot that folded its ledger, and the lab collector's own."""
        st = self.st or {}
        return {"ended": dict(self._fetch_ends()), "cluster": copy.deepcopy(st.get("cluster_fetched") or {}),
                "lab": self._lab_fetched()}

    def _fetch_ends(self):
        """{proposal id: utc} of this campaign's fetch attempts (FETCH_LEVERS)
        whose end the lane observed (state fetch_ends, written by _done and
        _failed). A state written before the record existed takes them once
        from the campaign ledger's item_done and failed entries, so attempts
        that ended before it count their fetched bytes too."""
        st = self.st
        if not isinstance(st, dict):
            return {}
        if not isinstance(st.get("fetch_ends"), dict):
            ends = {}
            try:
                with open(str(self.paths.ledger), "r", encoding="utf-8") as fh:
                    for line in fh:
                        if '"item_done"' not in line and '"failed"' not in line:
                            continue
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        if rec.get("campaign") == self.name and rec.get("event") in ("item_done", "failed") \
                                and rec.get("lever") in FETCH_LEVERS and rec.get("proposal_id"):
                            ends[str(rec["proposal_id"])] = rec.get("utc")
            except OSError:
                pass
            st["fetch_ends"] = ends
        return st["fetch_ends"]

    def _lab_fetched(self):
        """The fetched facts (_fetch_facts) of the lab collector's own ledger
        (lab INC_DIR/intake/sources.jsonl: L16L and L16RL append there, and no
        snapshot folds it; a fetch that failed after finishing some files
        records those), or None when it cannot be read whole (missing,
        unreadable, a line that is not JSON): the lab's ended fetches then
        count their requested max_bytes."""
        from . import stream_remote as SR
        rows = []
        try:
            with open(str(self.paths.lab_inc / "intake" / "sources.jsonl"), "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if isinstance(r, dict) and r.get("source"):
                        rows.append(r)
        except (OSError, ValueError):
            return None
        rows.sort(key=lambda r: (str(r.get("ts") or ""), str(r.get("source") or "")))
        ff = SR.fetch_facts(rows)
        return None if ff is None else _fetch_facts(ff)

    # ---- entry
    def go(self):
        st0 = load_state(self.paths, self.name)
        if st0 is None:
            if not self.cfg.get("enabled"):
                return {"enabled": False, "mode": "stream"}
            st0 = blank_state(self.name, self.cfg)
        self.st = copy.deepcopy(st0)
        for ln in ALL_LANES:                         # a state written before a lane existed
            self.st["lanes"].setdefault(ln, {"phase": "IDLE", "item": None, "fails": 0, "hold": None,
                                             "diag_hold": None, "until_utc": None})
        try:
            out = self._step()
            self.st["errors"] = 0
            self.st["last_error"] = None
            self.st["last_tick_utc"] = self.st["updated_utc"] = self.utc
            _write_json(self.paths.state(self.name), self.st)
            self._mirror_config()
            return out
        except Exception as e:
            self.st = st0
            st0["errors"] = int(st0.get("errors") or 0) + 1
            st0["last_error"] = {"utc": self.utc, "error": "%s: %s" % (type(e).__name__, _short(e)),
                                 "trace": _short(traceback.format_exc(), 3000)}
            st0["last_tick_utc"] = self.utc
            self._ledger("error", error=st0["last_error"]["error"])
            if st0["errors"] >= int(LS.t(self.th or LS.load_thresholds(), "stop_loss", "errors_to_pause")):
                self._pause("the ticker raised on %d ticks in a row (last: %s)" % (st0["errors"],
                                                                                   st0["last_error"]["error"]))
            try:
                _write_json(self.paths.state(self.name), st0)
            except Exception:
                pass
            self.log.warning("[inc-stream] %s: tick raised %s" % (self.name, st0["last_error"]["error"]))
            return {"error": st0["last_error"]["error"], "mode": "stream"}

    def _mirror_config(self):
        from . import campaign as C
        st = self.st
        view = {"mode": "stream", "lanes": {ln: {"phase": st["lanes"][ln]["phase"],
                                                  "item": (st["lanes"][ln].get("item") or {}).get("lever"),
                                                  "hold": st["lanes"][ln].get("hold") or st["lanes"][ln].get("diag_hold")}
                                             for ln in ALL_LANES},
                "card": ({"kind": st["card"].get("kind"), "title": st["card"].get("title")} if st.get("card") else None),
                "paused": (st.get("paused") or {}).get("reason")}
        cur = self.cfg.get("state") if isinstance(self.cfg.get("state"), dict) else {}
        if {k: cur.get(k) for k in view} == view:
            return

        def change(c):
            c["state"] = dict(view, updated_utc=self.utc)
            return c
        try:
            C._update_config(self.cfg_hooks, self.name, change)
        except Exception as e:
            self.log.warning("[inc-stream] %s: config view not written: %s" % (self.name, e))

    def _step(self):
        st = self.st
        st["ticks"] = int(st.get("ticks") or 0) + 1
        self._seed_ended()
        self._resumed()
        self._reopen_sources()
        why = self._paused_reason()
        if why:
            return {"mode": "stream", "paused": why, "ssh": False}
        bad = check_stream_config(self.cfg)
        if bad:
            self._card("config", "The stream campaign's config cannot run", "; ".join(bad))
            return {"mode": "stream", "error": "; ".join(bad)}
        self.dom = LS.load_domain((self.cfg.get("stream") or {}).get("stream_domain") or self.domain)
        self.th = LS.load_thresholds()
        self.prior = DS.prior_evidence(self.dom)
        self._log_decisions()
        env_end = _secs(self.cfg.get("envelope_end_utc"))
        if env_end is not None and self.now > env_end:
            self._pause("envelope_ended: the campaign envelope ran to %s; a person sets a new one (card X14)"
                        % self.cfg.get("envelope_end_utc"))
            return {"mode": "stream", "paused": "envelope_ended", "ssh": False}
        before = self.ssh.calls
        if not self.paths.latest(self.name).is_file() or self.st.get("await_snapshot"):
            # nothing observed yet (or only the stream a fork replaced): this
            # tick's one call is the snapshot
            self._poll_lab()
            self._adopt_lost_runs()
            self._observe()
            return {"mode": "stream", "ssh": self.ssh.calls > before, "first": True}
        # 1. lab work on the cached evidence (no ssh)
        self._poll_lab()
        self._adopt_lost_runs()
        self._intake_retry_sweep()
        self._build_evidence()
        self._diagnose()
        if self._paused_reason():
            return {"mode": "stream", "paused": self._paused_reason(), "ssh": self.ssh.calls > before}
        self._materialise()
        self._run_lab_items()
        # 2. the one ssh: the ready items, else the snapshot
        if self.ssh.left() and not self._paused_reason():
            if not self._submit_ready():
                self._observe()
        return {"mode": "stream", "ssh": self.ssh.calls > before,
                "lanes": {ln: {"phase": st["lanes"][ln]["phase"], "item": (st["lanes"][ln].get("item") or {}).get("lever"),
                               "status": (st["lanes"][ln].get("item") or {}).get("status")} for ln in ALL_LANES}}

    def _log_decisions(self):
        """DEC-A..DEC-D (human) and L-1..L-7 (human-delegated), logged once per campaign."""
        st = self.st
        if st.get("decisions_logged"):
            return
        for d in self.dom.get("decisions") or []:
            self._ledger("decision", decided_by=d.get("decided_by"), id=d.get("id"), text=d.get("text"),
                         decided_utc=d.get("utc"), domain_config_sha256=self.dom.get("_sha256"))
        st["decisions_logged"] = True

    # ---- the lab's own files
    def _collect_config(self):
        rel = (self.cfg.get("stream") or {}).get("collect_config") or self.dom.get("collect_config")
        if not rel:
            return None, None
        p = Path(rel)
        if not p.is_absolute():
            p = LS.TOOLS_DIR / rel
        return p, _read_json(p)

    def _never_train(self):
        return LS.never_train_slugs(self.dom)

    def _candidate_row(self, source):
        """The raw row of a source in the lab's candidates.json, or None."""
        raw = _read_json(self.paths.candidates())
        rows = raw.get("candidates") if isinstance(raw, dict) else raw
        for r in rows or []:
            if isinstance(r, dict) and str(r.get("source_id") or r.get("id") or "") == str(source):
                return r
        return None

    def _candidates(self):
        """The lab's candidates (L15's candidates.json, collect-candidates/1) and
        the known items of the collect config, normalised, with each source's
        state merged in. The collector's own pre-check (`precheck`: failures
        with their action and the governance class of the decision they ask
        for) is honoured: a close or refuse action refuses the source, an R3
        hold files it for a person, another hold waits."""
        raw = _read_json(self.paths.candidates())
        rows = raw.get("candidates") if isinstance(raw, dict) else raw
        rows = [r for r in rows or [] if isinstance(r, dict)]
        _p, cc = self._collect_config()
        known = [dict(r, known_item=True) for r in ((cc or {}).get("known_items") or []) if isinstance(r, dict)]
        placement = ((self._artifact("intake/placement.json") or {}).get("providers")) or {}
        lab_only = set((((cc or {}).get("placement") or {}).get("lab_only")) or [])
        overrides = (cc or {}).get("licence_overrides")
        overrides = overrides if isinstance(overrides, dict) else {}
        nt, nt_err = self._never_train()
        by = {}
        for r in known + rows:
            cid = str(r.get("source_id") or r.get("id") or r.get("source") or r.get("slug") or "")
            if not cid:
                continue
            cur = by.get(cid, {})
            cur.update({k: v for k, v in r.items() if v is not None})
            cur["id"] = cid
            cur["known_item"] = bool(cur.get("known_item") or r.get("known_item"))
            by[cid] = cur
        out = []
        for cid in sorted(by):
            r = by[cid]
            classes = r.get("target_classes")
            if classes is None:
                classes = [c.get("name") for c in r.get("classes_mapped") or []
                           if isinstance(c, dict) and c.get("status") in ("target", "target_synonym")]
            lic = r.get("licence", r.get("license"))
            lic_ok = r.get("licence_ok")
            lcls = lic.get("class") if isinstance(lic, dict) else None
            if isinstance(lic, dict):
                if lic_ok is None and lic.get("class") in ("refused",):
                    lic_ok = False
                lic = lic.get("id") if lic.get("class") not in (None, "unresolved") else None
            if lic is None and lic_ok is True and r.get("licence_override"):
                lic = r.get("licence_id")        # a person's licence override, as collect.plan records it
            # a person's licence override the collect config records, read here as _fold reads it: a candidates
            # file plan wrote before the override still says unresolved, and its pre-check still holds
            # licence_unresolved, until the next L15; an unresolved licence only, never a refused one
            ov = overrides.get(cid)
            by_ov = isinstance(ov, dict) and bool(ov.get("id")) and lcls == "unresolved" and lic_ok is not False
            if by_ov and lic is None:
                lic, lic_ok = ov.get("id"), True
            lifted = ("licence_unresolved",) if by_ov else ()
            s = (self.st.get("sources") or {}).get(cid) or {}
            prov = r.get("provider")
            pr = (placement.get(prov) or {}) if isinstance(placement, dict) else {}
            on_cluster = pr.get("placement") == "cluster" if "placement" in pr else pr.get("status") == "pass"
            pl = "lab" if prov in lab_only or pr.get("lab_only") or not on_cluster else "cluster"
            pc = r.get("precheck") if isinstance(r.get("precheck"), dict) else {}
            fails = [f for f in pc.get("failures") or [] if isinstance(f, dict)]
            names_pending = bool(r.get("names_pending")) or (r.get("decision") or {}).get("status") == "pending_names" \
                or any(f.get("code") == "names_pending" for f in fails)
            out.append({"id": cid, "provider": prov, "licence": lic, "licence_ok": lic_ok,
                        "target_classes": sorted(set(c for c in classes or [] if c)),
                        "bytes": r.get("bytes"), "images": r.get("images"),
                        "expected_target_boxes": r.get("expected_target_boxes"), "target_boxes": r.get("target_boxes"),
                        "lab_group": r.get("lab_group"), "evaluation_lab": bool(r.get("evaluation_lab")),
                        "copy_scan_done": bool(r.get("copy_scan_done")),
                        "credentials": r.get("credentials_ok", r.get("credentials")),
                        "image_level": bool(r.get("image_level") or r.get("annotation_type") in ("image-level",
                                                                                                "image_level")
                                            or r.get("annotation") == "image_level"),
                        "never_train": (nt is None) or cid in (nt or set()), "known_item": r["known_item"],
                        "status": s.get("status"), "partial": bool(s.get("partial")), "placement": pl,
                        "in_shards": DS.mid_shards(s),
                        # its fetch is complete on the cluster and its intake failed since: L16I again, never a
                        # new fetch (_intake_retry_sweep)
                        "intake_retry": bool(DS.fetch_on_cluster(s) and self._intake_failed_after_fetch(cid, s)),
                        "attempts": int(s.get("attempts") or 0), "failures": int(s.get("failures") or 0),
                        "names_unresolved": (bool(r.get("names_unresolved")) or names_pending)
                        and not s.get("names_resolved"),
                        "names_resolved": bool(s.get("names_resolved")),
                        "collector_precheck": {"refuse": sorted(f.get("code") for f in fails
                                                                if f.get("action") in ("close", "refuse")
                                                                and f.get("code") != "names_pending"),
                                               "review": sorted(f.get("code") for f in fails
                                                                if f.get("action") == "hold" and f.get("risk") == "R3"
                                                                and f.get("code") not in lifted),
                                               "wait": sorted(f.get("code") for f in fails
                                                              if f.get("action") == "hold" and f.get("risk") != "R3"
                                                              and f.get("code") != "names_pending"
                                                              and f.get("code") not in lifted)}})
        if nt is None:
            self._once("never_train_unreadable", nt_err, "never_train_unreadable", reasons=[nt_err])
            self._card("data", "The never-train list cannot be read: no source is collected", nt_err)
        return out

    # ---- evidence
    def _artifact(self, name):
        p = _read_json(self.paths.latest(self.name)) or {}
        arts = (((p.get("stream") or {}).get("decision") or {}).get("artifacts")) or {}
        return arts.get(name)

    def _record(self, payload):
        """The campaign-snapshot-shaped record evidence.from_snapshot reads: the
        live experiments' decision parts, the frozen ones, and the stream summary
        with the cached dev scores merged in. display_only never enters it."""
        exps = {}
        for exp, sub in ((payload or {}).get("experiments") or {}).items():
            snap = (sub or {}).get("snapshot")
            if isinstance(snap, dict) and snap.get("built", True) is not False:
                exps[exp] = {"snapshot": {k: v for k, v in snap.items() if k != "display_only"}}
        for exp in sorted(self.st.get("frozen") or {}):
            if exp in exps:
                continue
            fz = _read_json(self.paths.frozen(self.name, exp))
            if isinstance(fz, dict) and isinstance(fz.get("snapshot"), dict):
                exps[exp] = {"snapshot": fz["snapshot"]}
        stream = copy.deepcopy((payload or {}).get("stream") or {"verb": "stream-summary", "decision": {"artifacts": {}}})
        stream["verb"] = "stream-summary"
        stream.setdefault("decision", {}).setdefault("artifacts", {})
        # the tick's clock at which the snapshot was taken, not the cluster's wall
        # clock: the evidence's bytes (and the digest) must not change with it
        return {"verb": "campaign-snapshot", "utc": self.st.get("last_snapshot_utc"), "experiments": exps,
                "stream": stream}

    def _build_evidence(self):
        payload = _read_json(self.paths.latest(self.name)) or {}
        ctx = self._context(payload)
        self._ctx, self._payload = ctx, payload          # D22 read again with TRAIN idle (_cut_still_called)
        try:
            self.ev = E.from_snapshot(self._record(payload), self.sid, context=ctx,
                                      domain=LS.funnel_ref(self.dom))
        except (E.EvidenceError, KeyError, TypeError, ValueError) as e:
            self._once("evidence_error", str(e), "evidence_error", error=_short(e, 500))
            self.ev = E.from_texts({}, self.sid, context=ctx, domain=LS.funnel_ref(self.dom))

    def _stage(self, payload):
        """The rollout's state as the diagnoses read it (context /stage): the
        lock, every tracked experiment's status, the baselines, Stage A and C,
        the verdicts the platform has run, Step 1's one-time jobs."""
        st = self.st
        r0 = (st.get("stage") or {}).get("r0") or {}
        lock = self._artifact("splits/v2/lock_status.json") or {}
        status = {x.get("exp"): x for x in (((payload or {}).get("status") or {}).get("experiments") or [])
                  if isinstance(x, dict)}
        exp_status = {e: ("done" if x.get("done") else "built" if x.get("built") else "missing")
                      for e, x in sorted(status.items()) if e}
        base = {}
        for b in (self.dom.get("baselines") or {}).get("items") or []:
            # building while the platform's build runs; failed once a measurement arm's build ended without its
            # experiment (record only: DR0 does not propose it again; a person rebuilds it or leaves it)
            x = exp_status.get(b["exp"])
            mine = r0.get("baseline_%s" % b["id"])
            base[b["id"]] = x if x in ("done", "built") else (mine if mine in ("building", "failed") else "missing")
        sa_exp = (self.dom.get("stage_a") or {}).get("exp")
        sc_exp = "%s_c001" % self.sid
        s1 = (self._artifact("step1_stream/status.json") or {}).get("one_time") or {}
        # a measurement arm's native-resolution rescore (L23N): done once its record says complete, else what
        # the platform ran (running, done, failed), else missing
        native = {}
        for b in (self.dom.get("baselines") or {}).get("items") or []:
            if b.get("measure"):
                rec = self._artifact("%s/native_rescore.json" % b["exp"]) or {}
                native[b["id"]] = "done" if rec.get("status") == "complete" else (
                    r0.get("native_%s" % b["id"]) or "missing")
        # E1 (2026-10-03): splits v3 (L23V) is done once its summary says complete, else what the platform ran;
        # E1's agnostic rescore (L23E) once its record says complete
        b3 = self._artifact("splits/v3/summary.json") or {}
        if b3.get("status") == "complete":
            base3 = "done"
        elif b3.get("status"):
            base3 = str(b3["status"])           # over_walltime (or any other): no E1 arm builds on it
        elif r0.get("base3") == "done":
            base3 = "unconfirmed"               # its job ended, but no complete summary.json was shipped
        else:
            base3 = r0.get("base3") or "missing"
        e1b = self._e1_exp("B")
        ag = (self._artifact("%s/agnostic_rescore.json" % e1b) or {}) if e1b else {}
        agnostic = "done" if ag.get("status") == "complete" else (r0.get("agnostic") or "missing")
        # E2 (2026-10-04): its rescore and verdict (L23C) once capacity/e2_rescore.json says complete, else what
        # the platform ran
        e2r = self._artifact((self.dom.get("e2") or {}).get("rescore_record") or "capacity/e2_rescore.json") or {}
        e2 = "done" if e2r.get("status") == "complete" else (r0.get("e2") or "missing")
        # E2-C (2026-10-04, later): its attribution rescore and record (L23D) once capacity/e2_attr_rescore.json
        # says complete, else what the platform ran
        ear = self._artifact((self.dom.get("e2_attr") or {}).get("rescore_record")
                             or "capacity/e2_attr_rescore.json") or {}
        e2_attr = "done" if ear.get("status") == "complete" else (r0.get("e2_attr") or "missing")
        zoo = self._zoo_stage(payload, r0)
        # E3 (2026-10-05): an arm's scores (L23F) are done once capacity/e3_score_<arm>.json says complete, else
        # what the platform ran; E3's verdict (L23G) once capacity/e3_rescore.json says complete
        e3d = self.dom.get("e3") or {}
        e3_score = {}
        for x in e3d.get("order") or []:
            rec = self._artifact((e3d.get("score_records") or {}).get(x) or "capacity/e3_score_%s.json" % x) or {}
            e3_score[x] = "done" if rec.get("status") == "complete" else (r0.get("e3_score_%s" % x) or "missing")
        e3r = self._artifact(e3d.get("rescore_record") or "capacity/e3_rescore.json") or {}
        e3 = "done" if e3r.get("status") == "complete" else (r0.get("e3") or "missing")

        def state_of(exp, key):
            x = exp_status.get(exp)
            return x if x in ("done", "built") else ("building" if r0.get(key) == "building" else "missing")
        return {"lock": bool(lock.get("locked")), "splits_built": bool(r0.get("splits_built")),
                "train_manifests": lock.get("train_manifests") or [], "baselines": base, "native": native,
                "base3": base3, "agnostic": agnostic, "e2": e2, "e2_attr": e2_attr, "zoo": zoo,
                "e3_score": e3_score, "e3": e3, "v3fam": self._v3fam_stage(exp_status, payload),
                "ds1": self._ds1_stage(exp_status, payload),
                "exp_status": exp_status, "verdicts": dict(r0.get("verdicts") or {}),
                "protocol_v3_accepted": bool(self.cfg.get("protocol_v3_accepted_by")),
                "stage_a": {"exp": sa_exp, "status": state_of(sa_exp, "stage_a")},
                "stage_a_ready": bool((st.get("stage_a") or {}).get("ready")) or (self._artifact("%s/%s" % (
                    sa_exp, (self.dom.get("stage_a") or {}).get("record") or "stage_a.json")) or {}).get("status")
                == "READY",
                "stage_c": {"exp": sc_exp, "status": state_of(sc_exp, "stage_c")},
                "stage_c_submitted": bool(r0.get("stage_c_built")),
                "stage_c_decided": bool(st.get("stage_c")), "capacity": (st.get("capacity") or {}).get("chosen"),
                "placement": bool(self._artifact("intake/placement.json")), "probe_ran": bool(r0.get("probe")),
                "step1_stream": {k: bool(s1.get(k)) for k in ("bootstrap", "knowntruth", "backfill")},
                "prospective": bool(st.get("prospective"))}

    def _context(self, payload):
        st = self.st
        q = self._artifact("stream/%s/queue_summary.json" % self.sid) or {}
        Mv = int(q.get("M") or (self.dom.get("increment") or {}).get("M") or 0)
        pool = q.get("pool") or {}
        bud = None
        try:
            bud = X.budget_now(self.camp, self.xctx)
        except Exception:
            bud = None
        al = {}
        prj = (payload or {}).get("projects") or {}
        pat = re.compile(str((self.dom.get("allocation") or {}).get("resource_pattern") or "GPU"), re.I)
        blk = [r for r in prj.get("resources") or [] if isinstance(r, dict) and pat.search(str(r.get("resource")))]
        al["balance_su"] = blk[0].get("balance_su") if blk else None
        al["end_date"] = (blk[0].get("end_date") if blk and blk[0].get("end_date") else
                          (self.dom.get("allocation") or {}).get("end_date"))
        al["reserve_su"] = self.cfg.get("alloc_reserve_su")
        # what every campaign has committed and not yet spent: the balance the
        # `projects` output reports already has the spent SU taken off, so
        # spent SU is not subtracted a second time (D27)
        al["committed_all_su"] = (bud or {}).get("domain_committed_su")
        dl = {}
        try:
            from ..brain import su_ledger as SL
            r = SL.remaining(self.domain, B.domain_budget(self.domain_budget)[0], base_dir=self.xctx.su_base_dir)
            dl = {k: r.get(k) for k in ("campaign_su", "envelope", "remaining_su")}
        except Exception as e:                       # unknown is never "enough": reported, not assumed
            dl = {"error": "%s: %s" % (type(e).__name__, _short(e, 200))}
        qu = dict((payload or {}).get("quota") or {})
        qu["staging_gb"] = sum((_num(s.get("bytes")) or 0.0) for s in (st.get("sources") or {}).values()
                               if s.get("status") in ("fetched", "names_pending", "intaken", "shard_pending")) / 1e9
        lanes = {}
        for ln in LANES:
            lane = st["lanes"][ln]
            lanes[ln] = {"phase": lane["phase"], "busy": lane.get("item") is not None,
                         "hold": lane.get("hold") or lane.get("diag_hold"), "half_cadence": bool(lane.get("half"))}
        arm = self._arm()
        builds = [_num(v) for v in (st.get("build_hours") or [])]
        return {"now_utc": self.utc, "sid": self.sid, "M": Mv, "K_max": LS.t(self.th, "D22", "K_max"),
                "lanes": lanes, "queue": {"Q": 0}, "candidates": self._candidates(),
                "discover": dict(st.get("discover") or {}), "sources": copy.deepcopy(st.get("sources") or {}),
                "refusals": list(st.get("refusals") or []),
                "budget": {k: (bud or {}).get(k) for k in ("envelope_su", "spent_su", "committed_su", "remaining_su",
                                                          "window_remaining_su", "domain_remaining_su")},
                "allocation": al, "quota": qu, "collect_gb_envelope": self.cfg.get("collect_gb_envelope"),
                "domain_ledger": dl,
                "stage": self._stage(payload), "segments": copy.deepcopy(st.get("segments") or []),
                "milestones": copy.deepcopy(st.get("milestones") or []),
                "milestone0": self._milestone0(),
                "pool": {"images": pool.get("images") or (self.dom.get("increment") or {}).get("base_images"),
                         "id": pool.get("current")},
                "arm": arm, "recipes": [r for r in self._recipes().split(",") if r],
                "build_hours_max": max([b for b in builds if b is not None] or [None])
                if [b for b in builds if b is not None] else None,
                "d33_history": list(st.get("d33_history") or []),
                "d33_species": sorted({s for h in st.get("d33_history") or [] if h.get("fired")
                                       for s in h.get("species") or []}),
                "doublings": int(st.get("doublings") or 0), "bisected": st.get("bisected"),
                # DR0's bounded wait for a person before the base v3 build (_lift_wait); outside /stage, which
                # some R0 items cite whole
                "lift_wait": copy.deepcopy(((st.get("stage") or {}).get("r0") or {}).get("lift_wait")),
                "rollbacks": copy.deepcopy(st.get("rollbacks") or []),
                # the label audits (L4) that ran, by experiment: D31 reads their records and never calls for L4 on
                # an experiment again ({exp: {utc, proposal_id, refreeze}}; refreeze: the frozen record's re-take)
                "audits": {e: {"utc": a.get("utc"), "proposal_id": a.get("proposal_id"),
                               "refreeze": ((st.get("refreeze") or {}).get(e) or {}).get("state")}
                           for e, a in sorted(self._audits_run().items())},
                "limits": {"gb_per_source": (LS.limits("L16") or {}).get("gb_per_source"),
                           "attempts_per_source": (LS.limits("L16") or {}).get("attempts_per_source")}}

    def _milestone0(self):
        """Milestone 0: the stream's own (choose-arm adopts the capacity
        decision's chosen experiment), else the default arm's baseline."""
        q = self._artifact("stream/%s/queue_summary.json" % self.sid) or {}
        rec = (((q.get("milestones") or {}).get("records") or {}).get("0")) or {}
        if rec.get("exp"):
            return rec["exp"]
        cap = self.dom.get("capacity") or {}
        return ((cap.get("arms") or {}).get(cap.get("default_arm")) or {}).get("exp")

    def _arm(self):
        """The capacity arm the stream runs ({name, cost_factor, exp}): the
        capacity decision's chosen arm once recorded, else the default."""
        cap = self.dom.get("capacity") or {}
        name = (self.st.get("capacity") or {}).get("chosen") or cap.get("default_arm")
        return dict(((cap.get("arms") or {}).get(name) or {}), name=name)

    # ---- diagnoses, stop-losses, holds
    def _diagnose(self):
        st = self.st
        diags = DS.detect(self.ev, self.dom, self.th, prior=self.prior)
        self.diags = diags + list(st.get("health") or [])
        fired = [d for d in diags if d.get("fired")]
        _write_json(self.paths.diagnoses(self.name), {"utc": self.utc, "sid": self.sid, "diagnoses": diags,
                                                     "rules_version": DS.rules_version()})
        self._once("diagnosed", [(d["id"], d.get("summary")) for d in fired], "diagnosed",
                   fired=[{"id": d["id"], "name": d.get("name"), "summary": d.get("summary"),
                           "cites": d.get("cites")} for d in fired], rules_version=DS.rules_version(),
                   digest_sha256=self.digest())
        by = DS.by_id(diags)
        # stop-losses that pause
        for did in ("D10S", "D27"):
            d = by.get(did) or {}
            if d.get("fired") and "OP_PAUSE" in (d.get("levers") or []):
                if "X14" in (d.get("levers") or []):
                    self._card("budget", "Budget or allocation: %s" % _short(d.get("summary"), 120),
                               d.get("summary"), lever="X14", trigger=[did])
                return self._pause("%s (%s): %s" % ((d.get("detail") or {}).get("reason") or did, did,
                                                    _short(d.get("summary"), 400)))
        # DR0's bounded wait before the base v3 build, before the fired diagnoses' cards (an escalation raised
        # this tick then stays the current card, and the alarm with it)
        self._lift_wait(by)
        # E2's decided verdict (2026-10-04): one card naming each qualifying arm's test read, a person's step
        self._e2_card()
        # E2-C's decided attribution (2026-10-04, later): one card, and E2-C's test read beside the headline
        self._e2_attr_card()
        # the zoo's chain stopped part-way with nobody following it (a person's submit, a killed submit): one card
        self._zoo_stale_card()
        # E3's decided verdict (2026-10-05): one card, with each qualifying arm's test read, a person's step
        self._e3_card()
        # V3F (2026-10-11): the family's person steps, its waits, D5b's block, and its decided verdict with the
        # exact commands of every sealed-test read (each a card, raised once per content)
        self._v3fam_cards(by)
        # DS1 (2026-10-29): what waits for a person (each a card naming it, holding no lane), the storage waits, the
        # decided verdicts with the sealed reads' commands
        self._ds1_cards(by)
        # cards and holds
        for d in fired:
            for lv in d.get("levers") or []:
                if str(lv).startswith("X"):
                    card = LS.card(lv) or {}
                    title = card.get("title") or lv
                    if not card:
                        from . import levers as LV
                        title = ((LV.load_menu().get("cards") or {}).get(lv) or {}).get("title") or lv
                    self._card("research", "%s: %s" % (lv, title), d.get("summary"), lever=lv, trigger=[d["id"]])
            if "OP_ESCALATE" in (d.get("levers") or []) or "OP_CARD" in (d.get("levers") or []):
                self._card("escalation", "%s %s" % (d["id"], d.get("name")), d.get("summary"), trigger=[d["id"]])
        holds = {ln: None for ln in LANES}
        for did, hold_of in (("D26", None), ("D30", "TRAIN"), ("D32", "TRAIN"), ("D25", None), ("D33", None),
                             ("D22", None), ("D27", None), ("DCAN", "TRAIN"), ("D28", None)):
            d = by.get(did) or {}
            if not d.get("fired"):
                continue
            det = d.get("detail") or {}
            h = det.get("hold") if hold_of is None else hold_of
            if did == "D22" and det.get("refused"):
                h = "TRAIN"
            for ln in ([h] if isinstance(h, str) else list(h or [])):
                if ln in holds and holds[ln] is None:
                    holds[ln] = "%s %s" % (did, d.get("name"))
        # D33's history (consecutive segments) and D21's closures
        d33 = by.get("D33") or {}
        if d33.get("fired") and d33.get("exp") and not (d33.get("detail") or {}).get("prospective"):
            hist = [h for h in st.get("d33_history") or [] if h.get("exp") != d33["exp"]]
            hist.append({"exp": d33["exp"], "fired": True, "species": (d33.get("detail") or {}).get("species")})
            st["d33_history"] = hist[-12:]
        d28 = by.get("D28") or {}
        # the stream ledger's quarantines are the truth (2026-10-24): a source inc2.stream quarantined is recorded so
        # here, whatever an admit's done handler wrote over it (2026-10-06/07: L24 quarantined three D28 leaks, the
        # same tick's L17 admit set them back to 'admitted', the next tick held them 'quarantine pending' and
        # re-rendered the L24 that had run, and lane STOP stalled for 17 days)
        self._resync_quarantines()
        for h in ((d28.get("detail") or {}).get("leaks") or []) if d28.get("fired") else []:
            s = st["sources"].setdefault(str(h.get("source")), {})
            if s.get("status") not in ("closed", "quarantined", "held"):
                s.update(status="held", held_reason="source leak (D28): quarantine pending")
                self._ledger("source_held", source=h.get("source"), reasons=["D28 leak"], trigger=["D28"])
        d21 = by.get("D21") or {}
        for c in ((d21.get("detail") or {}).get("close") or []) if d21.get("fired") else []:
            s = st["sources"].setdefault(c["source"], {})
            if s.get("status") not in ("closed", "quarantined"):
                s.update(status="closed", closed_reason=c.get("why"), closed_utc=self.utc)
                self._ledger("source_closed", source=c["source"], reasons=[c.get("why")], trigger=["D21"])
        d31 = by.get("D31") or {}
        # the blames D21 closes on and L24 quarantines on: those committed after a person's last lift of the
        # source's quarantine (D31 counted; per_source, every blame, for a record without it)
        d31d = d31.get("detail") or {}
        counted = d31d.get("counted") if isinstance(d31d.get("counted"), dict) else None
        for src, n in ((d31d.get("per_source") or {}).items() if d31.get("fired") else []):
            st["sources"].setdefault(src, {})["blames"] = int(n if counted is None else counted.get(src) or 0)
        d27 = by.get("D27") or {}
        st["want_largest"] = bool(d27.get("fired") and (d27.get("detail") or {}).get("hold") == "DATA")
        if st["want_largest"] and (d27.get("detail") or {}).get("largest"):
            self._card("quota", "/ocean is short of space for collection: a person frees space",
                       "largest directories: %s" % ", ".join("%s %.0f GB" % (x.get("dir"), x.get("gb") or 0)
                                                            for x in (d27["detail"]["largest"] or [])[:10]),
                       trigger=["D27"])
        d29 = by.get("D29") or {}
        lane = st["lanes"]["DATA"]
        if d29.get("fired") and lane.get("item") is None and lane["phase"] != "WAIT_DATA":
            days = int((d29.get("detail") or {}).get("wait_days") or 7)
            lane.update(phase="WAIT_DATA", until_utc=_utc(self.now + days * 86400))
            self._card("wait_data", "Collection exhausted: waiting %d days for new sources" % days, d29.get("summary"),
                       trigger=["D29"])
            self._ledger("wait_data", days=days, until_utc=lane["until_utc"], trigger=["D29"])
        d32 = by.get("D32") or {}
        st["lanes"]["DATA"]["half"] = bool(d32.get("fired"))
        for ln in LANES:
            old = st["lanes"][ln].get("diag_hold")
            st["lanes"][ln]["diag_hold"] = holds[ln]
            if holds[ln] != old:
                self._ledger("lane_hold" if holds[ln] else "lane_hold_cleared", lane=ln, hold=holds[ln] or old)
        # R0 records: Stage A, Stage C, the capacity decision
        dsa = by.get("DSA") or {}
        if dsa.get("fired") and not (st.get("stage_a") or {}).get("ready"):
            self._record_stage_a(dsa)
        dsc = by.get("DSC") or {}
        if dsc.get("fired") and not st.get("stage_c"):
            st["stage_c"] = {k: v for k, v in (dsc.get("detail") or {}).items() if k != "cites"}
            self._ledger("stage_c", verdict=st["stage_c"], trigger=["DSC"])
        dcap = by.get("DCAP") or {}
        if dcap.get("fired") and not st.get("capacity"):
            st["capacity"] = dict(dcap.get("detail") or {})
            self._ledger("capacity_decision", decision=st["capacity"], trigger=["DCAP"],
                         reasons=["L-4, as inc2.baseline capacity-verdict recorded it (dev only): an arm qualifies "
                                  "when its dev mean beats n640's by more than 2 pooled sd"])
        held = [ln for ln in LANES if st["lanes"][ln].get("hold")]
        if len(held) >= int(LS.t(self.th, "stop_loss", "held_lanes_to_pause")):
            return self._pause("stop-loss: %d lanes are held (%s)" % (len(held), ", ".join(
                "%s: %s" % (ln, st["lanes"][ln]["hold"]) for ln in held)))

    def _lift_wait(self, by):
        """DR0's bounded wait before the base v3 build (diagnose_stream.
        lift_wait). The diagnoses are recomputed every tick, so the stream
        state keeps the first tick DR0 deferred L23V (stage.r0.lift_wait,
        read back as context /lift_wait): the bound runs from there, and a
        card is raised from that tick on. The wait's sources follow D28's
        judged list; a tick on which D28 cannot judge (or the evidence
        cannot be read) changes nothing. It ends when DR0 finds the list
        empty on a readable queue summary (lifted), or once base v3 is no
        longer missing (running, done, failed); past the bound it is marked
        expired and kept, so it never starts again. Called before the
        fired diagnoses' cards, so an escalation raised the same tick stays
        the current card."""
        st = self.st
        r0 = st.setdefault("stage", {}).setdefault("r0", {})
        rec = r0.get("lift_wait") if isinstance(r0.get("lift_wait"), dict) else None
        if rec and r0.get("base3") not in (None, "missing"):
            r0.pop("lift_wait", None)
            self._ledger("lift_wait_ended", sources=rec.get("sources"), first_seen_utc=rec.get("first_seen_utc"),
                         trigger=["DR0"], reasons=["base v3 is %s" % r0.get("base3")])
            return
        det = (by.get("DR0") or {}).get("detail") or {}
        w, end = det.get("lift_wait"), det.get("lift_wait_end")
        if isinstance(w, dict) and w.get("state") == "waiting":
            if rec is None:
                rec = r0["lift_wait"] = {"first_seen_utc": w.get("first_seen_utc") or self.utc,
                                         "sources": w.get("sources")}
                self._ledger("lift_wait", sources=w.get("sources"), first_seen_utc=rec["first_seen_utc"],
                             until_utc=w.get("until_utc"), basis=w.get("basis"), trigger=["DR0", "D28"],
                             reasons=["base v3 (L23V) waits for a person to lift the stream's quarantine of sources "
                                      "D28 now judges chance, or for D28 to judge them, bounded"])
            elif w.get("sources") is not None:
                rec["sources"] = w["sources"]
            self._lift_card(w)
        elif isinstance(end, dict) and rec:
            if end.get("state") == "lifted":
                r0.pop("lift_wait", None)
                self._ledger("lift_wait_ended", sources=rec.get("sources"), first_seen_utc=rec.get("first_seen_utc"),
                             trigger=["DR0", "D28"], reasons=["D28 lists no cleared source the stream still "
                                                              "quarantines on its word"])
            elif end.get("state") == "expired" and not rec.get("expired_utc"):
                rec["expired_utc"] = self.utc
                self._ledger("lift_wait_expired", sources=rec.get("sources"), first_seen_utc=rec.get("first_seen_utc"),
                             until_utc=end.get("until_utc"), trigger=["DR0"],
                             reasons=["%g h passed: L23V is proposed with the quarantine as it stands"
                                      % float(end.get("hours") or 0)])

    def _lift_card(self, w):
        until, srcs = w.get("until_utc"), w.get("sources")
        if srcs:
            self._card("quarantine_lift",
                       "Base v3 (L23V) waits until %s: a person lifts the stream's quarantine of %s" % (
                           until, ", ".join(srcs)),
                       "D28 now judges the dHash hits of %s chance, but the stream still quarantines %s, and the base "
                       "v3 build (inc2.base3 build, L23V) reads the quarantine when its job starts: a quarantined "
                       "source never enters arm B. Only a person lifts a quarantine; on the cluster, from the "
                       "repository's root: %s (if inc2.stream answers that the stream is being written, run it again "
                       "a little later). L23V is proposed once they are lifted, or at %s (%g h after L23V was first "
                       "deferred, %s) with the quarantine as it stands: to keep a quarantine, do nothing."
                       % (", ".join(srcs), "it" if len(srcs) == 1 else "them", "; ".join(w.get("commands") or []),
                          until, float(w.get("hours") or 0), w.get("first_seen_utc")),
                       trigger=["DR0", "D28"])
        else:
            self._card("quarantine_lift",
                       "Base v3 (L23V) waits until %s: D28 cannot judge the quarantined sources" % until,
                       "D28 could not be judged (a rule error or an unreadable queue summary), so whether a source "
                       "the stream quarantines is now judged chance is unknown, and the base v3 build (L23V) reads "
                       "the quarantine when its job starts. L23V waits for D28 to judge again, at most until %s (%g h "
                       "after L23V was first deferred, %s), then is proposed with the quarantine as it stands; a "
                       "person looks at D28's diagnosis." % (until, float(w.get("hours") or 0),
                                                              w.get("first_seen_utc")),
                       trigger=["DR0", "D28"])

    def digest(self):
        """The sha256 of everything a decision read (the evidence's canonical
        bytes); test blindness (S12) keeps it identical when every test and
        non-decision exam value changes."""
        return hashlib.sha256(self.ev.canonical()).hexdigest() if self.ev is not None else None

    def _record_stage_a(self, dsa):
        st = self.st
        rec = {"format": "inc-autopilot/prospective-stage-a/1", "sid": self.sid, "ready": True,
               "detail": dsa.get("detail"), "rules_version": DS.rules_version(), "utc": self.utc,
               "cites": dsa.get("cites")}
        path = self.paths.replay / ("prospective_stage_a_%s__%s.json" % (self.sid, DS.rules_version()))
        _write_json(path, rec)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        st["stage_a"] = {"ready": True, "recipes": (dsa.get("detail") or {}).get("recipes"), "path": str(path),
                         "sha256": sha, "chosen": (dsa.get("detail") or {}).get("chosen")}
        self._ledger("stage_a_ready", path=str(path), sha256=sha, recipes=st["stage_a"]["recipes"],
                     survivors=(dsa.get("detail") or {}).get("survivors"), trigger=["DSA"])

    # ---- lanes: from diagnoses to items
    def _train_ready(self):
        """"" when the TRAIN lane may cut a segment (R0 READY, R0b and R1), else why not."""
        stg = self._stage(_read_json(self.paths.latest(self.name)) or {})
        entries = [e for e in (self._artifact("stream/%s/ledger.jsonl" % self.sid) or []) if isinstance(e, dict)]
        led = [e.get("event") for e in entries]
        miss = []
        # a commit (L19) shows in the evidence only at the next snapshot, and a
        # tick that submits takes no snapshot: the next segment is not cut
        # until the diagnoses that read the last commit (D8S, D30, D32, D33,
        # D31) have read it, or their hold would come one segment late
        seen = {e.get("exp") for e in entries if e.get("event") == "commit"}
        unseen = [s["exp"] for s in self.st.get("segments") or [] if s.get("committed") and s["exp"] not in seen]
        if unseen:
            miss.append("the commit of %s is not yet in the evidence (the next snapshot shows it)" % ", ".join(unseen))
        why = self._quarantine_unseen()
        if why:
            miss.append(why)
        if not stg["lock"]:
            miss.append("splits v2 not locked")
        m0 = self._milestone0()
        if m0 and stg["exp_status"].get(m0) != "done":
            miss.append("milestone 0 (%s) not done" % m0)
        can = next((b for b in (self.dom.get("baselines") or {}).get("items") or []
                    if b.get("verdict") == "canary-verdict"), None)
        if can and (self._artifact("%s/canary.json" % can["exp"]) or {}).get("passed") is not True:
            miss.append("the canary has not passed")
        if not stg["stage_a_ready"]:
            miss.append("Stage A not READY")
        chosen = (self.st.get("capacity") or {}).get("chosen")
        if not chosen:
            miss.append("no capacity decision (L-4)")
        elif self.ev is not None and ((self.ev.json(E.CONTEXT) or {}).get("arm") or {}).get("name") != chosen:
            # recorded this tick, after the evidence was built: the walltime and
            # price rules (D26) read the arm from the evidence, so the first cut
            # waits one tick for evidence that carries the chosen arm
            miss.append("the capacity decision is newer than this tick's evidence")
        if "init" not in led:
            miss.append("the stream is not created")
        elif "arm" not in led:
            miss.append("the stream has not adopted the capacity decision")
        if not self.st.get("stage_c"):
            miss.append("Stage C not decided")
        if not stg["step1_stream"].get("bootstrap"):
            miss.append("step1_stream not bootstrapped")
        return "; ".join(miss)

    def _quarantine_unseen(self):
        """Why a segment's cut waits for a quarantine (L24), or "". A
        quarantine takes its source's rows out of the eligible queue, so D22's
        Q is stale while one is pending in the STOP lane or ran after the last
        snapshot (a tick that submits takes no snapshot). On 2026-10-05 04:53Z
        L18 was proposed in the tick L24 quarantined zenodo_15808623, Q fell
        from 8,041 to 319 (M 1,364), and the cutter refused the build 'short'."""
        st = self.st
        it = st["lanes"]["STOP"].get("item") or {}
        if it.get("lever") == "L24":
            return "a quarantine (L24 on %s) is %s: the queue the cut reads changes with it" % (
                ((it.get("proposal") or {}).get("params") or {}).get("source"), it.get("status"))
        seen = str(st.get("last_snapshot_utc") or "")
        q = sorted(src for src, s in (st.get("sources") or {}).items()
                   if isinstance(s, dict) and s.get("quarantined_utc") and str(s["quarantined_utc"]) >= seen)
        if q:
            return "the quarantine of %s is not yet in the evidence (the next snapshot shows it)" % ", ".join(q)
        return ""

    def _cut_order(self, lever):
        """Why `lever` must wait for the other of the pair L18 (a segment's
        cut) and L23V (the base v3 build), or "". inc2.base3 reads the
        stream's pools and in-flight rows when its job starts and again
        before it writes (it refuses when a listed row was cut meanwhile),
        and the cutter refuses test v1 rows only once the lists exist: a
        segment is not submitted while base v3 is submitted and not
        finished (stage.r0.base3 running), and base v3 is not submitted while
        a submitted segment's cut is not yet in the evidence (the stream
        ledger's build line of its experiment), so neither job runs while
        the other's result is unknown to it. Proposed or filed, neither
        holds the other: the lanes' submission order decides."""
        st = self.st
        if lever == "L18":
            if ((st.get("stage") or {}).get("r0") or {}).get("base3") == "running":
                return "base v3 (L23V) is submitted and not finished: a segment cut now could take rows it holds out"
            return ""
        if lever == "L23V":
            it = st["lanes"]["TRAIN"].get("item") or {}
            if it.get("lever") != "L18" or it.get("status") in ("proposed", "filed"):
                return ""
            child = (it.get("proposal") or {}).get("child_exp")
            led = self._artifact("stream/%s/ledger.jsonl" % self.sid) or []
            if not any(isinstance(e, dict) and e.get("event") == "build" and e.get("exp") == child for e in led):
                return ("segment %s (L18) is submitted and its cut is not in the evidence yet: base v3 would not see "
                        "its rows" % child)
        return ""

    def _prospective_guard(self):
        """The prospective stream record of the current stream rules version
        (written once, before the first L18); "" when it holds."""
        st = self.st
        ver = DS.rules_version()
        rec = st.get("prospective") or {}
        q = self._artifact("stream/%s/queue_summary.json" % self.sid) or {}
        # one record per rules version AND per stream version: a fork (L22)
        # doubles M under a new sid, and its first L18 needs its own record
        if rec.get("rules_version") == ver and rec.get("sha256") and rec.get("sid", self.sid) == self.sid \
                and rec.get("M", q.get("M")) == q.get("M"):
            return ""
        body = DS.prospective_record(self.dom, self.th, self.cfg, {"stage_a": st.get("stage_a"),
                                                                  "capacity": st.get("capacity"),
                                                                  "M": q.get("M")}, self.utc)
        path = self.paths.replay / ("prospective_stream_%s__%s.json" % (self.sid, ver))
        if not path.is_file():
            _write_json(path, body)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        st["prospective"] = {"path": str(path), "sha256": sha, "rules_version": ver, "record_sha256": body["sha256"],
                             "sid": self.sid, "M": q.get("M")}
        self._ledger("prospective_stream", path=str(path), sha256=sha, rules_version=ver,
                     record_sha256=body["sha256"], M=body["M"], K_max=body["K_max"])
        return ""

    def _intents(self):
        """[(lane, lever, params, trigger id, extra)] from the fired diagnoses, in
        detect's order (stops and TRAIN first, then DATA, then MAINT's R0)."""
        out = []
        for d in self._fired():
            det = d.get("detail") or {}
            did = d["id"]
            props = det.get("propose")
            props = props if isinstance(props, list) else ([props] if isinstance(props, dict) else [])
            if did == "DR0":
                for ln, x in sorted((det.get("due") or {}).items()):
                    props.append(dict(x))
            if did == "DHOLD":
                for h in det.get("holds") or []:
                    props.append({"lever": h["lever"], "hold": h["hold"],
                                  **({"verb": "scan-holds"} if h["lever"] == "L17" else {})})
            for pr in props:
                lever = pr.get("lever")
                if lever in LANE_OF:
                    out.append((LANE_OF[lever], lever, pr, did, d))
        return out

    def _materialise(self):
        st = self.st
        if st.get("drift"):
            return
        self._adopt_person_items()
        # the lanes a diagnosis wants this tick, counted per intent: an intent whose proposal already ran and ended
        # done takes its count back (_done_recalled), so a lane wanted only for such work ends its stall
        self._stalls, want = {}, {}
        for lane_name, lever, pr, did, d in self._intents():
            if lever in PERSON_LEVERS:
                self._file_person(lane_name, lever, pr, did, d)
                continue
            lane = st["lanes"][lane_name]
            if lane.get("item") is not None or lane.get("hold") or lane.get("diag_hold"):
                continue
            want[lane_name] = want.get(lane_name, 0) + 1
            if lane_name == "DATA" and lane.get("half") and st["ticks"] % 2:
                continue
            if lane_name == "DATA" and lane["phase"] == "WAIT_DATA":
                if lever == "L15" and self.now < (_secs(lane.get("until_utc")) or 0):
                    continue
                if lever != "L15":
                    lane["phase"] = "IDLE"
            if lane_name == "TRAIN" and lever == "L18":
                why = self._train_ready()
                if why:
                    self._once("train_wait", why, "not_taken", lever="L18", reasons=["R0 not READY: " + why])
                    self._stall(lane_name, lever, did, "R0 not READY: " + why)
                    continue
                why = self._cut_order("L18")
                if why:
                    self._once("cut_order:L18", why, "not_taken", lever="L18", reasons=[why])
                    self._stall(lane_name, lever, did, why)
                    continue
                self._prospective_guard()
            try:
                p = self._proposal(lever, pr, did, d)
            except (LS.LeverError, ValueError, KeyError) as e:
                self._once("render:%s:%s" % (lever, did), str(e), "not_taken", lever=lever,
                           reasons=["cannot render %s: %s" % (lever, _short(e, 300))])
                self._stall(lane_name, lever, did, "cannot render %s: %s" % (lever, _short(e, 300)))
                continue
            if p is None:
                continue
            declined = p["id"] in (st.get("declined") or [])
            # an item that ran and ended done (state done_ids; for an id filed as declined before done_ids existed,
            # the campaign ledger's item_done). A lab item keeps its own path: its step may be wanted again and runs
            # under a new id (_adopt_declined, _recover_lab -> _renew)
            ran_at = self._done_at(p["id"], ledger=declined) if p.get("follow") != "lab" else None
            if declined and not ran_at:
                # a lab item declined after a restart lost its run: the run is taken back (its effect lives only
                # in this state); any other declined id is never proposed again, recorded once, never silently
                if p.get("follow") == "lab" and self._adopt_declined(lane_name, lever, p, did, d):
                    continue
                why = ("%s (proposal %s) was declined earlier and is never proposed again; %s calls for it on "
                       "every tick" % (p.get("lever") or lever, p["id"], did))
                self._once("declined:%s" % p["id"], p["id"], "not_taken", lane=lane_name, lever=p.get("lever") or lever,
                           proposal_id=p["id"], trigger=[did], reasons=[why])
                self._stall(lane_name, p.get("lever") or lever, did, why)
                continue
            if self._step_exhausted(lane_name, lane, p):
                continue
            done_at = (st.get("done_keys") or {}).get(_sha([p.get("lever") or lever, p.get("params") or {}])[:16])
            # done at or after the snapshot the evidence came from (an item the
            # fold of that snapshot finished may not show its effect in it)
            if done_at and done_at >= str(st.get("last_snapshot_utc") or ""):
                # done since the evidence was taken: the next snapshot shows its effect first
                self._once("stale:%s" % p["id"], done_at, "not_taken", lever=lever,
                           reasons=["%s ran at %s, after the last snapshot: waiting for a snapshot that shows it"
                                    % (lever, done_at)])
                self._stall(lane_name, lever, did, "%s ran at %s, after the last snapshot" % (lever, done_at))
                continue
            if ran_at:
                # the same id ran and ended done, and the snapshot since shows its effect: never run again, never a
                # stall (2026-10-05..26: D31's L4)
                self._done_recalled(lane_name, p.get("lever") or lever, p, did, ran_at)
                want[lane_name] -= 1
                continue
            lever = p.get("lever") or lever           # the sub-lever the proposal took (L16 -> L16L on the lab)
            lane["item"] = {"proposal": p, "lever": lever, "status": "proposed", "since_utc": self.utc,
                            "attempt": p.get("attempt", 0), "trigger": did, "diagnoses": [d]}
            lane["phase"] = self._phase_of(lever)
            self._ledger("proposed", lane=lane_name, lever=lever, action=p["policy_action"], risk=p["risk"],
                         argv=p["argv"], est_gpu_hours=p.get("est_gpu_hours"), trigger=[did],
                         cites=p.get("cites"), proposal_id=p["id"], child_exp=p.get("child_exp"))
        self._file_reviews()
        self._watchdog({ln for ln, n in want.items() if n > 0})

    def _stall(self, ln, lever, did, why):
        """A diagnosis's lever on an idle, unheld lane not taken this tick for
        a reason that can last (_watchdog counts such ticks). The waits by
        design are not: DATA's half cadence, WAIT_DATA's discovery date, and a
        proposal with nothing to act on (_proposal None: a source already
        quarantined, closed this tick)."""
        self._stalls.setdefault(ln, []).append({"lever": lever, "trigger": did, "why": _short(why, 400)})

    def _watchdog(self, wanted):
        """A card when a lane stalls: on watchdog stall_ticks ticks in a row a
        fired diagnosis called for a lever on the lane, nothing held it and
        nothing took it (_stall). Once per stall; a tick on which the lane holds
        an item, is held, or no diagnosis calls for it ends the stall, and a
        tick with only waits by design leaves the count as it is. Live, from
        2026-10-04 09:27Z: D20 called for L16 on rf_a-programlama__ag-programlama
        every tick and the DATA lane stood idle about 24 h, its L16S skipped as
        declined with no event and no card."""
        st = self.st
        n = int(LS.t(self.th, "watchdog", "stall_ticks"))
        stalls = st.setdefault("stalls", {})
        for ln in ALL_LANES:
            lane = st["lanes"][ln]
            got = self._stalls.get(ln)
            if lane.get("item") is not None or lane.get("hold") or lane.get("diag_hold") or ln not in wanted:
                old = stalls.pop(ln, None)
                if old and old.get("carded"):
                    self._ledger("lane_stall_ended", lane=ln, since_utc=old.get("since_utc"), ticks=old.get("ticks"))
                continue
            if not got:
                continue
            rec = stalls.setdefault(ln, {"since_utc": self.utc, "ticks": 0})
            rec["ticks"] = int(rec.get("ticks") or 0) + 1
            rec["levers"] = sorted(set(rec.get("levers") or []) | {g["lever"] for g in got})
            rec["triggers"] = sorted(set(rec.get("triggers") or []) | {g["trigger"] for g in got})
            rec["last"] = got[-1]["why"]
            if rec["ticks"] < n or rec.get("carded"):
                continue
            rec["carded"] = self.utc
            title = "Lane %s stalled: %s called for %s on %d ticks and nothing took the lane" % (
                ln, ", ".join(rec["triggers"]), ", ".join(rec["levers"]), rec["ticks"])
            detail = ("Since %s every tick a fired diagnosis called for %s on lane %s, nothing held the lane (no "
                      "stop-loss, no diagnosis hold) and no item took it. The last reason: %s. A person reads it: a "
                      "declined proposal is never proposed again (stream ledger: not_taken, recovered, refused); a "
                      "render error or a wait that does not end is a defect." % (
                          rec["since_utc"], ", ".join(rec["levers"]), ln, rec["last"]))
            self._card("stall", title, detail, lever=rec["levers"][0], trigger=rec["triggers"])
            self._ledger("lane_stalled", lane=ln, since_utc=rec["since_utc"], ticks=rec["ticks"], levers=rec["levers"],
                         trigger=rec["triggers"], reasons=[rec["last"]])

    @staticmethod
    def _phase_of(lever):
        return {"L15": "DISCOVER", "L26": "NAMES", "LP": "PROBE", "L16": "COLLECT", "L16L": "COLLECT",
                "L16R": "COLLECT", "L16RL": "COLLECT", "L16S": "SYNC", "L16I": "INTAKE", "L17": "ADMIT",
                "L24": "QUARANTINE", "LH": "HOLD_RELEASE", "L18": "SEGMENT", "L19": "COMMIT", "L22": "FORK",
                "L20": "MILESTONE", "L21": "ROLLBACK", "L23": "SPLITS", "L23B": "BASELINE", "L23N": "NATIVE",
                "L23V": "BASE3", "L23E": "AGNOSTIC", "L23C": "E2_VERDICT", "L23D": "E2_ATTR", "L23Z": "ZOO",
                "L23F": "E3_SCORE", "L23G": "E3_VERDICT", "L23H": "V3FAM_DATA", "L23I": "V3FAM_BUILD",
                "L23J": "V3FAM_SCORE", "L23K": "V3FAM_DETR", "L23L": "V3FAM_VERDICT",
                "L30": "DS1_DATA", "L31": "DS1_SHEETS", "L32": "DS1_BUILD", "L32X": "DS1_LONG", "L33": "DS1_SCORE",
                "L34": "DS1_VERDICT",
                "L25": "STAGE_A", "L27": "BISECT",
                "L28": "STAGE_C", "L4": "AUDIT", "LV": "VERDICT", "LI": "INIT", "LA": "ARM",
                "LC": "COMPARE"}.get(lever, "ITEM")

    def _failed_ids(self):
        """Proposal ids of this campaign's lane items that ended failed (state
        failed_ids). A state written before the list existed takes them once
        from the campaign ledger's 'failed' entries, so a step that failed
        before this record existed is not proposed under its old id either."""
        st = self.st
        if not isinstance(st.get("failed_ids"), list):
            ids = []
            try:
                with open(str(self.paths.ledger), "r", encoding="utf-8") as fh:
                    for line in fh:
                        if '"failed"' not in line:
                            continue
                        try:
                            rec = json.loads(line)
                        except ValueError:
                            continue
                        if rec.get("campaign") == self.name and rec.get("event") == "failed" \
                                and rec.get("proposal_id") and rec["proposal_id"] not in ids:
                            ids.append(rec["proposal_id"])
            except OSError:
                pass
            st["failed_ids"] = ids[-FAILED_IDS_KEEP:]
        return list(st["failed_ids"])

    def _step_exhausted(self, ln, lane, p):
        """True, with lane `ln` held for a person, when the step of proposal p
        (step_key: its lever and policy params) has failed more than
        stop_loss step_retries times: it is proposed again at most that many
        times after a failure. A person's release (stream release, or
        enable) frees the lane and clears the lane's step counts. The
        collection levers are bounded per source instead (SOURCE_BOUND)."""
        if p.get("lever") in SOURCE_BOUND:
            return False
        key = step_key(p.get("lever"), p.get("params"))
        rec = (self.st.get("step_failures") or {}).get(key) or {}
        n = int(rec.get("n") or 0)
        retries = int(LS.t(self.th, "stop_loss", "step_retries"))
        if n <= retries:
            return False
        lane["hold"] = ("stop-loss: %s failed %d times (last: %s); a step is proposed again at most %d times after a "
                        "failure, so a person decides whether to run it again: stream release --name %s --by "
                        "human:<email>" % (p.get("lever"), n, _short(rec.get("last"), 200), retries, self.name))
        lane["hold_utc"] = self.utc
        self._card("stop_loss", "Lane %s held: %s failed %d times" % (ln, p.get("lever"), n), lane["hold"],
                   lever=p.get("lever"))
        self._ledger("lane_held", lane=ln, hold=lane["hold"], lever=p.get("lever"), step=key,
                     proposal_id=rec.get("proposal_id"))
        return True

    def _release_steps(self, ln):
        """A person released lane `ln`: its steps' failure counts start again."""
        sf = self.st.get("step_failures") or {}
        for k in [k for k, v in sf.items() if (v or {}).get("lane") == ln]:
            sf.pop(k, None)

    def _next_name(self, kind):
        taken = set(self.st.get("exps") or [])
        n = 1
        while "%s_%s%03d" % (self.sid, kind, n) in taken:
            n += 1
        return "%s_%s%03d" % (self.sid, kind, n)

    def _proposal(self, lever, pr, did, d):
        """The lane item's proposal for lever with the diagnosis's params."""
        st, dom = self.st, self.dom
        cites = list(d.get("cites") or [])
        q = self._artifact("stream/%s/queue_summary.json" % self.sid) or {}
        arm = self._arm()
        info = {"pool_images": (self.ev.json(E.CONTEXT) or {}).get("pool", {}).get("images"), "M": q.get("M"),
                "arm": arm.get("name")}
        params, child, parent, lid = {}, None, None, lever
        floor_attempt = 0                         # a step's attempt starts here at least (L24 after a person's lift)
        pkg = self.pkg
        lab = str(self.paths.lab_inc)
        if lever == "L15":
            cp, _cc = self._collect_config()
            params = {"config": str(cp), "classes": ",".join(pr.get("classes") or []),
                      "out": str(self.paths.candidates())}
        elif lever == "L16":
            src = pr["source"]
            s = st["sources"].get(src) or {}
            if s.get("status") in ("closed", "quarantined", "held", "names_pending", "shard_pending") \
                    or DS.mid_shards(s):
                return None                       # closed this tick (D21, D28, a retry-false refusal), or its shards
            if int(s.get("attempts") or 0) >= int((LS.limits("L16") or {}).get("attempts_per_source") or 3):
                # a 4th attempt on one source is a loop, not a retry (S15)
                self._pause("stop-loss: a 4th collection attempt on source %s was about to be proposed" % src)
                return None
            params = {"source": src, "max_bytes": self._byte_plan(src, pr.get("bytes"))}
            if pr.get("placement") == "lab":
                lid = "L16L"
                params["out"] = str(self.paths.staging()) + "/"
            elif s.get("names_resolved") and not s.get("names_synced"):
                # its names were resolved on the lab (L26): the cluster's
                # collector reads them offline, so the layer is pushed first
                lid = "L16S"
                params = {"source": src, "names": 1}
            elif not pr.get("known_item"):
                # a discovered source fetched on the cluster: the collector there
                # reads its candidate record from the file the lab synced first
                if not s.get("candidate_synced"):
                    lid = "L16S"
                    params = {"source": src, "candidate": 1}
                else:
                    params["candidates"] = s["candidate_synced"]
        elif lever == "L26":
            params = {"source": pr["source"], "out": str(self.paths.names_dir()) + "/"}
        elif lever == "L16S":
            params = {"source": pr["source"]}
            if pr.get("names"):
                params["names"] = 1
        elif lever == "L16I":
            params = {"source": pr["source"]}
        elif lever == "L17":
            params = {"verb": pr.get("verb") or "admit"}
            if pr.get("intake"):
                params["intake"] = pr["intake"]
                imgs = ((self.ev.json("intake/%s/summary.json" % pr["intake"]) or {}).get("images"))
                info["images"] = imgs
            if pr.get("hold"):
                params["hold"] = pr["hold"]
            files = (((_read_json(self.paths.latest(self.name)) or {}).get("stream") or {}).get("files")) or {}
            ver = (self.ev.json("step1_stream/status.json") or {}).get("versions") or {}
            pins = {"verifier": ver.get("verifier"),
                    "lock_sha256": ver.get("splits_lock_sha256") or (files.get("splits/v2/lock_status.json") or {})
                    .get("sha256")}
            first = st.setdefault("pins", {})
            for k, v in pins.items():
                if v and first.get(k) and first[k] != v:
                    self._card("research", "X11: %s changed (%s -> %s)" % (k, first[k][:12], v[:12]),
                               "a verifier refit or a LOCK change is a versioned event (R4): no batch is admitted "
                               "under it until a person decides", lever="X11", trigger=[did])
                    return None
                first.setdefault(k, v)
            info["pins"] = pins
        elif lever == "L24":
            params = {"pkg": pkg, "source": pr["source"], "cite": pr.get("cite") or did}
            src_st = st["sources"].get(pr["source"]) or {}
            if src_st.get("status") == "quarantined" or src_st.get("leak_kept_by") \
                    or (src_st.get("blame_kept_by") and params["cite"] == "D31"):
                return None
            if str(pr["source"]) in self._stream_quarantined():
                # the stream's ledger already quarantines it: nothing to run; this state follows the ledger
                self._resync_quarantines()
                return None
            # a quarantine a person lifted (inc2.stream unquarantine) and the diagnosis calls for again (D31: blames
            # committed after the lift) is new work: each lift starts a new id, so the one that ran is never recalled
            floor_attempt = len([e for e in self._artifact("stream/%s/ledger.jsonl" % self.sid) or []
                                 if isinstance(e, dict) and e.get("event") == "unquarantine"
                                 and str(e.get("source")) == str(pr["source"])])
        elif lever == "LH":
            params = {"pkg": pkg, "stream": self.sid, "hold": pr.get("hold") or "funnel_F9"}
        elif lever == "L18":
            child = pr.get("exp") or q.get("next_segment") or self._next_name("s")
            params = {"pkg": pkg, "stream": self.sid, "k": int(pr.get("k") or 1), "exp": child}
            recipes = self._recipes()
            M_ = int(q.get("M") or dom["increment"]["M"])
            every = LS.truth_every(dom, self.th, int(info["pool_images"] or dom["increment"]["base_images"]), M_,
                                   tuple(recipes.split(",")), float(arm.get("cost_factor") or 1.0))
            # the stream's own truth cadence (L-4, per segment) decides; this is the price's
            info.update(recipes=recipes, truth_every=every)
        elif lever == "L19":
            params = {"pkg": pkg, "exp": pr["exp"]}
        elif lever == "L22":
            params = {"pkg": pkg, "stream": self.sid, "m": int(pr["m"])}
        elif lever == "L20":
            child = pr.get("exp") or ((q.get("milestones") or {}).get("next")) or self._next_name("m")
            params = {"pkg": pkg, "stream": self.sid}
        elif lever == "L21":
            params = {"pkg": pkg, "stream": self.sid, "to": pr.get("to") or "P_0"}
        elif lever == "L27":
            child = self._next_bisect_arm()
            params = {"pkg": pkg, "stream": self.sid, "from_pool": pr.get("from_pool") or "P_0"}
            info["n_suspect"] = pr.get("n_suspect")
        elif lever == "L28":
            if not q.get("M"):
                return None                       # the stream (and its M) does not exist yet
            child = "%s_c001" % self.sid
            params = {"pkg": pkg, "stream": self.sid, "holdout": (dom.get("stage_c") or {}).get("holdout"),
                      "m": int(q["M"])}
            info["recipes"] = ",".join(q.get("stage_b") or ["r0"])
        elif lever == "L23":
            params = {"pkg": pkg, "verb": pr.get("verb") or "build"}
        elif lever == "L23B":
            b = next(x for x in (dom.get("baselines") or {}).get("items") or [] if x["id"] == pr["baseline"])
            params = {"pkg": pkg, "exp": b["exp"], "seeds": b["seeds"], "arm": b.get("arm") or "n640",
                      "role": b["role"]}
            if b.get("union"):
                params["union"] = ",".join(LS.inc_path(x) for x in b["union"])
            else:
                params["manifest"] = LS.inc_path(b["manifest"])
            if b.get("e2"):
                params["e2"] = b["e2"]                   # E2's arm (2026-10-04): the build derives init and recipe
            child = b["exp"]
            info["images"] = b.get("images") or (dom.get("increment") or {}).get("base_images")
            info["arm"] = b.get("arm")
            if isinstance(b.get("budget"), dict):
                info["budget"] = dict(b["budget"])       # E1's cold_budget and E2's arms: priced from their image-epochs
        elif lever == "L23N":
            b = next(x for x in (dom.get("baselines") or {}).get("items") or []
                     if x["id"] == pr["baseline"] and x.get("measure"))
            ref = ((dom.get("capacity") or {}).get("native") or {})["reference_exp"]
            params = {"pkg": pkg, "exp": b["exp"], "reference": ref}
            parent = b["exp"]                     # it builds nothing: the job is followed by its id
            seeds = [x for x in str(b.get("seeds") or "").split(",") if x != ""]
            info["runs"] = 2 * len(seeds) if seeds else None
        elif lever == "L23V":
            params = {"pkg": pkg, "stream": self.sid}
        elif lever == "L23E":
            eb, ea = self._e1_exp("B"), self._e1_exp("A")
            if not (eb and ea):
                return None
            params = {"pkg": pkg, "exp": eb, "reference": ea}
            parent = eb                           # it builds nothing: the job is followed by its id
            n = sum(len([x for x in str(b.get("seeds") or "").split(",") if x != ""])
                    for b in self._e1_items().values())
            info["runs"] = n or None
        elif lever == "L23C":
            arms = self._e2_items()
            if sorted(arms) != ["S", "W"] or not all(arms.values()):
                return None
            params = {"pkg": pkg}
            ref = (dom.get("e2") or {}).get("reference_exp")
            rb = next((x for x in (dom.get("baselines") or {}).get("items") or [] if x["exp"] == ref), {})

            def nseeds(x):
                return len([s_ for s_ in str(x.get("seeds") or "").split(",") if s_ != ""])
            # one scoring pass per E2 final run (six) and per reference final run whose file may be missing (three)
            info["runs"] = sum(nseeds(b) for v_ in arms.values() for b in v_) + nseeds(rb)
        elif lever == "L23D":
            arms = self._e2_attr_items()
            if sorted(arms) != ["C", "W"] or not all(arms.values()):
                return None
            params = {"pkg": pkg}
            ref = (dom.get("e2") or {}).get("reference_exp")
            rb = next((x for x in (dom.get("baselines") or {}).get("items") or [] if x["exp"] == ref), {})

            def nseeds(x):
                return len([s_ for s_ in str(x.get("seeds") or "").split(",") if s_ != ""])
            # one scoring pass per E2-C final run (three) and per E2-W and reference final run whose file may be
            # missing (six), as L23C's are priced
            info["runs"] = sum(nseeds(b) for v_ in arms.values() for b in v_) + nseeds(rb)
        elif lever == "L23Z":
            z = dom.get("zoo") or {}
            if not z.get("record"):
                return None
            params = {"version": z["version"], "shards_a": int(z["shards_a"]), "shards_c": int(z["shards_c"]),
                      "concurrency": int(z["concurrency"]), "max_gpu_hours": int(z["max_gpu_hours"])}
        elif lever == "L23F":
            e3 = dom.get("e3") or {}
            order = list(e3.get("order") or [])
            if pr.get("arm") not in order:
                return None
            params = {"pkg": pkg, "arm": pr["arm"]}
            # its three arm-seeds (dev and the reported passes); the first arm's job also fits the
            # classifier, takes the dev ground-truth accuracy and runs the equivalence check
            info["runs"] = len(e3.get("seeds") or [0, 1, 2])
            info["fit"] = pr["arm"] == order[0]
        elif lever == "L23G":
            params = {"pkg": pkg}
        elif lever in V3FAM_LEVERS:
            got = self._v3fam_params(lever, pr, pkg, info)
            if got is None:
                return None
            params, child = got
        elif lever in DS1_LEVERS:
            got = self._ds1_params(lever, pr, pkg, info)
            if got is None:
                return None
            params, child = got
        elif lever == "L25":
            sa = dom.get("stage_a") or {}
            params = {"pkg": pkg, "exp": sa["exp"], "from_exp": sa["from_exp"], "recipes": sa["recipes"]}
            child, parent = sa["exp"], sa["from_exp"]
        elif lever == "LV":
            params = {"pkg": pkg, "module": pr["module"], "verb": pr["verb"]}
            if pr.get("exp"):
                params["exp"] = pr["exp"]
            info["verdict_key"] = pr.get("key")
        elif lever == "LI":
            params = {"pkg": pkg, "stream": self.sid, "stage_b": pr["stage_b"]}
        elif lever == "LA":
            params = {"pkg": pkg, "stream": self.sid}
        elif lever == "LC":
            params = {"pkg": pkg, "exp": pr["exp"]}
            parent = pr["exp"]
        elif lever == "LP":
            params = {}
        elif lever == "L4":
            return self._audit_proposal(pr, did, d)
        else:
            return None
        r = LS.row(lid)
        est = None
        if r.get("estimator") not in ("zero",):
            est, estimate = LS.price(lid, params, dom, info)
        else:
            estimate = {"estimator": "zero"}
        if lid in V3FAM_LEVERS and not self._v3fam_cap_ok(lid, params, est):
            return None
        if lid in DS1_LEVERS and not self._ds1_cap_ok(lid, params, est, pr):
            return None
        akey = step_key(lid, LS.policy_params(lid, params))
        attempt = max(int((st.get("attempts") or {}).get(akey) or 0), floor_attempt)
        failed_ids = set(self._failed_ids())
        for _ in range(64):
            p = LS.proposal(self.name, lid, params, trigger=[did], cites=cites, lane=LANE_OF.get(lever),
                            parent_exp=parent, child_exp=child, est=est, estimate=estimate, attempt=attempt)
            if p["id"] not in failed_ids:
                break
            # the id of a lane item that ended failed is never proposed again: the
            # executor runs an id once and refuses a changed proposal under an id
            # it filed (seen live on 2026-09-28: L23 after its job failed)
            attempt += 1
        p["lever"] = lid
        if info.get("pins"):
            p["pins"] = info["pins"]
        if info.get("verdict_key"):
            p["verdict_key"] = info["verdict_key"]
        if lid in DS1_LEVERS:
            p["ds1_item"] = pr.get("item") or self._ds1_key(lid, params)
            p["ds1_phase"] = int(pr.get("phase") or 1)
        if lever == "L18":
            p["base_pool"] = {k: (q.get("pool") or {}).get(k) for k in ("current", "sha256", "images")}
        return p

    def _recipes(self):
        """The recipes the next segment runs (pricing only; inc2.stream
        decides): the chosen recipe once Stage B chose one, else the stream's
        Stage B arms (init's --stage-b, Stage A's recorded recipes)."""
        q = self._artifact("stream/%s/queue_summary.json" % self.sid) or {}
        if q.get("chosen_recipe"):
            return str(q["chosen_recipe"])
        return ",".join(q.get("stage_b") or []) or str((self.st.get("stage_a") or {}).get("recipes") or "r0")

    def _next_bisect_arm(self):
        """The first arm the next bisect build makes: <sid>_bNNN, numbered
        across rollbacks after every arm the stream ledger records."""
        n = 0
        for e in self._artifact("stream/%s/ledger.jsonl" % self.sid) or []:
            if isinstance(e, dict) and e.get("event") == "bisect" and e.get("phase") == "build":
                n += len(e.get("arms") or {})
        return "%s_b%03d" % (self.sid, n + 1)

    def _byte_plan(self, src, est):
        """The max_bytes of a fetch of source `src` (L16 and its forms, L16R): the candidate's size estimate (the
        domain's collect.max_bytes_default, one 10 GiB shard, when it has none), or the larger plan a refusal raised
        it to (sources.<src>.byte_plan, _source_refused), never over the per-source cap (L16's gb_per_source). An
        estimate can be short of what is fetched: a GitHub repository's API size is its git object store, while the
        provider downloads the branch's zip, whose size is not listed (2026-10-07: gh_geezacoleman__palmer_detection
        was planned at its 64,145,408-byte estimate, its zip was 84,379,800 bytes, and both attempts were refused
        over_byte_plan). Such a refusal raises the plan once (_source_refused); the estimate stays the first plan, so
        what an attempt reserves, and what an attempt whose bytes are unknown counts (executor.fetched_bytes), is
        not inflated for every source."""
        cap = int(float((LS.limits("L16") or {}).get("gb_per_source") or 50) * 1e9)
        default = int(_num(((self.dom or {}).get("collect") or {}).get("max_bytes_default")) or cap)
        raised = int(_num(((self.st.get("sources") or {}).get(str(src)) or {}).get("byte_plan")) or 0)
        return max(1, min(cap, max(int(_num(est) or default), raised)))

    def _audit_proposal(self, pr, did, d):
        """L4 (the existing label audit) on a segment's data-blamed increments."""
        from . import levers as LV
        exp = pr["exp"]
        try:
            args, cites = LV.audit_args(self.ev, exp)
        except Exception as e:
            self._once("l4:%s" % exp, str(e), "not_taken", lever="L4", reasons=[_short(e, 300)])
            return None
        steps = set(pr.get("steps") or [])
        audits = [a for a in args["audits"] if a.split("=", 1)[0] in steps] or args["audits"]
        params = {"trusted": args["trusted"], "audit": ",".join(audits), "out": args["out"]}
        argv = ["sbatch", "run_inc_audit.sh", "--trusted", args["trusted"], "--audit"] + audits + ["--out", args["out"]]
        pid = _sha([self.name, "L4", argv])[:32]
        return {"id": pid, "lever": "L4", "family": "L4", "argv": argv, "params": params,
                "policy_action": "inc_label_audit", "risk": "R2", "trigger": [did],
                "cites": list(d.get("cites") or []) + cites, "lit": [], "control": "", "success": "", "falsifier": "",
                "est_gpu_hours": 0.0, "proposed_by": AUTO, "lane": "MAINT", "follow": "job",
                "parent_exp": exp, "child_exp": exp, "attempt": 0}

    def _context_candidate(self, src):
        """The source's candidate row in the evidence context (_candidates), or {}."""
        cands = ((self.ev.json(E.CONTEXT) if self.ev is not None else None) or {}).get("candidates") or []
        return next((c for c in cands if isinstance(c, dict) and c.get("id") == src), {})

    def _file_reviews(self):
        """D20's candidates that failed a pre-check: an R3 item for a person
        each, filed once, never taking a lane (6.3 L16). Its form follows the
        candidate's placement, as D20's L16 -> L16L does: L16R fetches on the
        cluster, L16RL on the lab, where the cluster's collector refuses a
        provider placement.json does not place on compute nodes
        (not_placed_on_cluster: job 47302914 ran an approved L16R of a
        lab-placed source). A review superseded because its cluster form met a
        lab-placed source (_misplaced_review), or because its cluster form
        already ran for a source now placed on the lab (_spent_cluster_review),
        is filed again, in the form its placement calls for."""
        st = self.st
        d20 = DS.by_id(self.diags).get("D20") or {}
        for r in ((d20.get("detail") or {}).get("review") or [])[:3]:
            src = r.get("source")
            if not src:
                continue
            prev = (st.get("source_reviews") or {}).get(src)
            cand = self._context_candidate(src)
            self._spent_cluster_review(src, prev, cand)
            superseded = isinstance(prev, dict) and prev.get("status") == "superseded"
            if prev is not None and not superseded:
                continue
            params = {"source": src, "max_bytes": self._byte_plan(src, cand.get("bytes"))}
            lid = "L16R"
            if cand.get("placement") == "lab":
                lid = "L16RL"
                params["out"] = str(self.paths.staging()) + "/"
            try:
                est, estimate = LS.price(lid, params, self.dom)
                p = LS.proposal(self.name, lid, params, trigger=["D20"], cites=d20.get("cites") or [],
                                est=est, estimate=estimate)
            except LS.LeverError as e:
                self._once("review:%s" % src, str(e), "not_taken", lever=lid, reasons=[str(e)])
                continue
            p["reason"] = "pre-check failed for %s: %s; a person decides" % (src, "; ".join(r.get("reasons") or []))
            res = X.submit(p, actor=AUTO, campaign=self.camp, ctx=self.xctx)
            rec = {"approval_id": res.get("approval_id"), "status": res.get("status"), "reasons": r.get("reasons"),
                   "utc": self.utc, "lever": lid}
            if superseded:
                rec["superseded"] = list(prev.get("superseded") or []) + [
                    {"approval_id": prev.get("approval_id"), "lever": prev.get("lever") or "L16R",
                     "utc": prev.get("superseded_utc"), "reason": prev.get("superseded_reason")}]
            st.setdefault("source_reviews", {})[src] = rec
            if res.get("status") == "filed" and res.get("approval_id"):
                # an approval by a person is adopted into the DATA lane and run
                self._park("DATA", p, "D20", d20, res.get("approval_id"))
            self._ledger("review_filed", lever=lid, source=src, approval_id=res.get("approval_id"),
                         status=res.get("status"), reasons=r.get("reasons"))
            if any("credentials" in x for x in r.get("reasons") or []):
                self._card("research", "X16: credentials for %s" % src, "; ".join(r["reasons"]), lever="X16",
                           trigger=["D20"])

    def _spent_cluster_review(self, src, prev, cand):
        """A source's filed review in its cluster form (L16R) whose approval
        already ran, for a source whose candidate is now placed on the lab, is
        marked superseded, so that _file_reviews files its lab form (L16RL)
        when D20 lists the source again: the cluster's collector refuses such a
        provider (not_placed_on_cluster), and one approval runs once. The live
        record: approval ap-1790700040-42144765 ran as job 47302914 before the
        lab form existed, and its review stayed 'filed'. D20 lists no source
        that is being fetched, so a run still in progress is never superseded."""
        if not isinstance(prev, dict) or prev.get("status") != "filed" or (prev.get("lever") or "L16R") != "L16R" \
                or cand.get("placement") != "lab":
            return
        a = AP.state(self.domain, root=self.xctx.approvals_root).get(prev.get("approval_id")) or {}
        ex = a.get("execution") or {}
        if ex.get("phase") not in ("done", "failed"):
            return
        why = ("its cluster form (L16R, approval %s) already ran (%s by %s, outcome %s), and its provider %s is "
               "placed on the lab: the cluster's collector refuses it (not_placed_on_cluster), so the review is "
               "filed again in its lab form (L16RL)"
               % (prev.get("approval_id"), ex.get("phase"), ex.get("executed_by"),
                  (ex.get("outcome") or {}).get("status"), cand.get("provider")))
        prev.update(status="superseded", superseded_utc=self.utc, superseded_reason=_short(why, 500))
        self._ledger("review_superseded", lever="L16R", source=src, approval_id=prev.get("approval_id"),
                     reasons=[why])

    # ---- lab items (no ssh)
    def _lab_hooks(self):
        run = self.runner
        target = os.environ.get("CLUSTER_SSH", "")

        def hook(kind):
            def h(params):
                job = "%s_%s_%s" % (kind, re.sub(r"[^A-Za-z0-9_]", "_", str(params.get("source") or "")),
                                    _sha([params, self.utc])[:10])
                if kind == "sync" and params.get("candidate"):
                    rel = self.paths.cand_rel(params["source"])
                    row = self._candidate_row(params["source"])
                    if row is None:
                        return {"ok": False, "error": "no candidate record of %s in the lab's candidates.json"
                                                      % params["source"]}
                    _write_json(self.paths.lab_inc / rel, {"format": "collect-candidates/1", "candidates": [row],
                                                           "from": str(self.paths.candidates()),
                                                           "written_utc": self.utc})
                    argv = sync_argv(self.paths.lab_inc, params["source"], target, M.CLUSTER_DATA_SSH, file=rel)
                elif kind == "sync":
                    argv = sync_argv(self.paths.lab_inc, params["source"], target, M.CLUSTER_DATA_SSH,
                                     names=bool(params.get("names")))
                else:
                    if kind == "names":
                        self._write_pending_names(params.get("source"))
                    verb = {"discover": "plan", "names": "names", "fetch": "fetch"}[kind]
                    # the lab's INC tree, named: the ticker's environment sets no INC_DIR, and inc.common's
                    # default is the cluster's /ocean path (2026-09-29: two L15 plans failed at intake_lock
                    # with PermissionError '/ocean' after five minutes of discovery)
                    argv = [sys.executable, "-m", "weed_optimizer_framework.tools.collect", verb,
                            "--inc-dir", str(self.paths.lab_inc)]
                    for k, flag in (("config", "--config"), ("classes", "--classes"), ("source", "--source"),
                                    ("max_bytes", "--max-bytes"), ("out", "--out")):
                        if params.get(k) not in (None, ""):
                            argv += [flag, str(params[k])]
                try:
                    got = run.launch(job, argv, timeout=lab_job_timeout(kind, params.get("max_bytes")))
                except Exception as e:
                    return {"ok": False, "error": "%s: %s" % (type(e).__name__, _short(e, 300))}
                return {"ok": True, "detached": True, "lab_job": got.get("job", job), "argv": argv}
            return h
        return {"inc_stream_discover": hook("discover"), "inc_stream_names": hook("names"),
                "inc_stream_collect_lab": hook("fetch"), "inc_stream_collect_review_lab": hook("fetch"),
                "inc_stream_sync": hook("sync"), "inc_ds1_sheets": self._ds1_sheets_hook}

    def _run_lab_items(self):
        """Start every ready lab item (a detached process; no ssh): a proposed
        one, and a filed one as _ready treats a filed cluster item (approved:
        run it; run elsewhere: follow that run; denied: clear it; a gated data
        lever under data_autonomy on: re-check its gates). _ready skips lab
        actions, and this loop once took proposed items only, so an approved
        lab fetch (L16L) waited for ever: the DATA lane stood still from
        2026-09-29 to 2026-09-30 with its approval recorded."""
        ap = None
        for ln in ALL_LANES:
            it = self.st["lanes"][ln].get("item")
            if not it:
                continue
            p = it["proposal"]
            if p["policy_action"] not in X.LAB_ACTIONS:
                continue
            if it.get("status") == "proposed":
                res = X.submit(p, actor=AUTO, campaign=self.camp, ctx=self.xctx)
            elif it.get("status") == "filed" and it.get("approval_id"):
                if ap is None:
                    ap = AP.state(self.domain, root=self.xctx.approvals_root)
                a = ap.get(it["approval_id"]) or {}
                if a.get("status") == "approved" and a.get("execution") is None:
                    res = X.execute_approved(it["approval_id"], self.camp, self.xctx, invoked_by=AUTO,
                                             quiet_repeat=True)
                elif a.get("status") == "approved":
                    self._executed_elsewhere(ln, it, a)
                    continue
                elif a.get("status") == "denied":
                    self._ledger("denied", lane=ln, lever=it["lever"], approval_id=it["approval_id"],
                                 decided_by=a.get("decided_by") or "human")
                    self.st["declined"] = (list(self.st.get("declined") or []) + [p["id"]])[-200:]
                    self._person_denied(it, a)
                    self._clear(ln)
                    continue
                elif a.get("status") == "withdrawn":
                    self._approval_withdrawn(ln, it, a)
                    continue
                elif p["policy_action"] in X.GATED_R2_ACTIONS and self.cfg.get("data_autonomy") == "on":
                    res = X.submit(p, actor=AUTO, campaign=self.camp, ctx=self.xctx)
                else:
                    continue
            else:
                continue
            self._on_result(ln, it, res)

    def _poll_lab(self):
        for ln in ALL_LANES:
            it = self.st["lanes"][ln].get("item")
            if not it or it.get("status") != "running" or not it.get("lab_job"):
                continue
            res = self.runner.poll(it["lab_job"])
            if res is None:
                continue
            self._ledger("lab_job_finished", lane=ln, lever=it["lever"], lab_job=it["lab_job"], ok=res.get("ok"),
                         rc=res.get("rc"), seconds=res.get("seconds"))
            if res.get("ok"):
                self._done(ln, it, res)
                continue
            why = "the lab process %s ended %s: %s" % (it["lab_job"], res.get("rc"),
                                                       _short(res.get("stderr_tail") or res.get("error") or "", 300))
            ref = None if res.get("lost") else self._collector_refusal(it, res.get("rc"), self._collect_result(res))
            if ref is not None:
                self._source_refused(ln, it, ref, why)
            else:
                self._failed(ln, it, why, lost=bool(res.get("lost")))

    # ---- runs a lane lost (a tick that ran them and did not write its state)
    def _executions(self):
        """This campaign's execution records (the executor's log, folded: one
        record per run), read once a tick."""
        if self._execs is None:
            tag, rows = json.dumps(self.name), []
            try:
                with open(str(self.xctx.exec_log), "r", encoding="utf-8") as fh:
                    for line in fh:
                        if tag not in line or not line.lstrip().startswith("{"):
                            continue
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(r, dict) and r.get("campaign") == self.name:
                            rows.append(r)
            except OSError:
                return []
            self._execs = B.fold(rows)
        return self._execs

    @staticmethod
    def _lab_job_of(rec):
        """The lab job an executed lab action's run recorded (its hook's
        payload, executor._run_plans), or None."""
        if (rec or {}).get("status") != "executed":
            return None
        pay = ((rec.get("remote") or {}).get("payload")) or {}
        return pay.get("lab_job") if isinstance(pay, dict) else None

    def _adopt_lost_runs(self):
        """Runs of this campaign the lanes lost, taken back (_adopt_run). The
        executor writes a run's record before the ticker writes its state, so a
        restart (or a raising tick) between the two leaves an executed record
        no lane follows: on 2026-10-03 07:44Z an L16S candidate sync ran (lab
        job sync_rf_a_programlama__ag_programlama_c2ad2f813a, ok), the
        dashboard restarted before the state was written, and the sync's
        effect (candidate_synced) was never recorded. An executed record of a
        lane lever is adopted when its proposal id is in no lane (or in its
        lane still 'proposed', which takes it in place), no item ended with it
        (ended_ids, failed_ids, declined), no person's parked item holds it,
        and its run was not adopted before. Only records at or after the
        state's last successful write (updated_utc) are read: an earlier run
        was seen by a tick whose state was written. A lost run whose lane is
        busy waits for it (adopt_from_utc keeps the horizon). A state written
        before ended_ids existed takes them from the campaign ledger first
        (_seed_ended)."""
        st = self.st
        horizon = str(st.get("adopt_from_utc") or st.get("updated_utc") or "")
        if not horizon:
            return
        lanes = st["lanes"]
        held = {}
        for ln in ALL_LANES:
            it = lanes[ln].get("item") or {}
            if (it.get("proposal") or {}).get("id"):
                held[it["proposal"]["id"]] = (ln, it)
        gone = set(st.get("ended_ids") or []) | set(self._failed_ids()) | set(st.get("declined") or []) \
            | set(st.get("person_items") or {})
        adopted = set(st.get("adopted_runs") or [])
        waiting = []
        for rec in self._executions():
            pid, lever, ts = rec.get("proposal_id"), rec.get("lever"), str(rec.get("ts") or "")
            if rec.get("status") != "executed" or not pid or lever not in LANE_OF or lever in PERSON_LEVERS \
                    or ts < horizon or pid in gone or (rec.get("run_id") or pid) in adopted:
                continue
            if pid in held:
                ln, it = held[pid]
                if it.get("status") != "proposed":
                    continue                     # followed already: running, or filed with its approval
            else:
                ln = LANE_OF[lever]
                if lanes[ln].get("item") is not None:
                    waiting.append(ts)
                    continue
                it = {"proposal": self._proposal_of(rec), "lever": lever, "status": "proposed", "since_utc": self.utc,
                      "attempt": None, "trigger": (rec.get("trigger") or [None])[0], "diagnoses": []}
            self._adopt_run(ln, it, rec, "executed at %s and lost with the state of that tick (a restart or a raising "
                                         "tick before the state was written)" % ts)
            held[pid] = (ln, it)
            adopted.add(rec.get("run_id") or pid)
        if waiting:
            st["adopt_from_utc"] = min(waiting)
        else:
            st.pop("adopt_from_utc", None)

    def _ledger_ended(self):
        """({proposal id: utc}, {lab job: utc}) of the items the lanes followed
        to an end, from the campaign ledger (ENDED_EVENTS name the proposal,
        lab_job_finished the lab job; the latest utc of each), read once a
        tick; None when the ledger cannot be read."""
        if self._ended is None:
            pids, jobs, done = {}, {}, {}
            try:
                with open(str(self.paths.ledger), "r", encoding="utf-8") as fh:
                    for line in fh:
                        if not any('"%s"' % e in line for e in ENDED_EVENTS + ("lab_job_finished",)):
                            continue
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        if not isinstance(r, dict) or r.get("campaign") != self.name:
                            continue
                        if r.get("event") in ENDED_EVENTS and r.get("proposal_id"):
                            pids[str(r["proposal_id"])] = str(r.get("utc") or "")
                            if r.get("event") == "item_done":
                                done[str(r["proposal_id"])] = {"utc": str(r.get("utc") or ""), "lever": r.get("lever"),
                                                               "exp": r.get("child_exp")}
                            else:
                                # a later end that is not done (an id is run once, so this is a record's repair)
                                done.pop(str(r["proposal_id"]), None)
                        elif r.get("event") == "lab_job_finished" and r.get("lab_job"):
                            jobs[str(r["lab_job"])] = str(r.get("utc") or "")
            except FileNotFoundError:
                pass
            except OSError:
                return None
            self._ended = (pids, jobs)
            self._ended_done = done
        return self._ended

    def _done_at(self, pid, ledger=False):
        """The utc at which the lane item of proposal `pid` ended done (state done_ids; with `ledger`, else the
        campaign ledger's item_done line, for an item done before done_ids existed, which is then recorded in
        done_ids), or None. A ledger that cannot be read: None (the caller keeps its old path)."""
        rec = (self.st.get("done_ids") or {}).get(str(pid)) if isinstance(self.st.get("done_ids"), dict) else None
        if isinstance(rec, dict) and rec.get("utc"):
            return str(rec["utc"])
        if not ledger or self._ledger_ended() is None:
            return None
        rec = (self._ended_done or {}).get(str(pid))
        if not rec:
            return None
        self._note_done(str(pid), rec)
        return rec["utc"]

    def _note_done(self, pid, rec):
        """Record proposal `pid` as ended done ({utc, lever, exp}) in done_ids, the oldest dropped past
        DONE_IDS_KEEP."""
        dn = self.st.get("done_ids") if isinstance(self.st.get("done_ids"), dict) else {}
        dn.pop(pid, None)
        dn[pid] = {"utc": rec.get("utc"), "lever": rec.get("lever"), "exp": rec.get("exp")}
        while len(dn) > DONE_IDS_KEEP:
            dn.pop(next(iter(dn)))
        self.st["done_ids"] = dn

    def _done_recalled(self, ln, lever, p, did, done_at):
        """A diagnosis calls for proposal p, whose item already ran and ended done at `done_at`: not run again (an id
        runs once), not declined, not a stall, and the lane is not wanted for it. One not_taken event and one card
        per id name the diagnosis that keeps calling for a lever after it ran: it does not read that lever's result
        (a rule to fix), or the same work is wanted again under the same id (a lever whose id must change)."""
        why = ("%s (proposal %s) ran and ended done at %s; %s still calls for it on every tick: it is not run again "
               "(an id runs once) and the lane does not wait for it. %s does not read %s's result, or the same work is "
               "wanted again under the same id: a rule to fix, not a stall" % (lever, p["id"], done_at, did, did, lever))
        if self._once("done_recalled:%s" % p["id"], done_at, "not_taken", lane=ln, lever=lever, proposal_id=p["id"],
                      trigger=[did], done_recalled=True, done_utc=done_at, reasons=[why]) is not None:
            self._card("platform", "%s %s ran at %s and %s still calls for it: not run again" % (
                lever, p["id"][:12], done_at, did), why, lever=lever, trigger=[did])

    def _seed_ended(self):
        """A state written before ended_ids existed (the first tick after the
        deploy that added them) takes them once from the campaign ledger: the
        items that ended at or after the state's horizon (_adopt_lost_runs),
        by proposal id, or by the lab job their run recorded. Without this,
        every run of the last tick that wrote the state, held by no lane (an
        L24 done when it ran), read as lost, and its effect was applied a
        second time. A ledger that cannot be read: no run before this tick is
        taken back (the horizon moves to it)."""
        st = self.st
        if isinstance(st.get("ended_ids"), list):
            return
        horizon = str(st.get("adopt_from_utc") or st.get("updated_utc") or "")
        led = self._ledger_ended() if horizon else ({}, {})
        if led is None:
            st["ended_ids"] = []
            st["adopt_from_utc"] = self.utc
            self._ledger("ended_ids_seeded", ids=0, reasons=["the campaign ledger cannot be read: runs before %s are "
                                                             "not taken back" % self.utc])
            return
        pids, jobs = led
        ids = {p for p, u in pids.items() if u >= horizon}
        jobs = {j for j, u in jobs.items() if u >= horizon}
        if jobs:
            ids |= {r["proposal_id"] for r in self._executions() if r.get("proposal_id") and self._lab_job_of(r) in jobs}
        st["ended_ids"] = sorted(ids)[-ENDED_KEEP:]
        if horizon:
            self._ledger("ended_ids_seeded", ids=len(st["ended_ids"]), since_utc=horizon)

    def _run_followed(self, rec):
        """True when a lane followed run `rec` to its end: its proposal id
        left a lane (ended_ids), or the campaign ledger records its item's end
        or its lab job's (_ledger_ended). A ledger that cannot be read counts
        as followed: a run is never taken back twice."""
        pid = (rec or {}).get("proposal_id")
        if pid in (self.st.get("ended_ids") or []):
            return True
        led = self._ledger_ended()
        if led is None:
            return True
        job = self._lab_job_of(rec)
        return pid in led[0] or bool(job and job in led[1])

    def _renew(self, ln, lever, p, rec, how):
        """The step of proposal p runs again under a new id (the executor runs
        an id once): its earlier run was followed to its end, so it is neither
        taken back nor declined. The step's attempt count rises past p's."""
        key = step_key(lever, p.get("params"))
        att = self.st.setdefault("attempts", {})
        att[key] = max(int(att.get(key) or 0), int(p.get("attempt") or 0)) + 1
        self._ledger("not_taken", lane=ln, lever=lever, proposal_id=p["id"], run_id=(rec or {}).get("run_id"),
                     lab_job=self._lab_job_of(rec), attempt=att[key],
                     reasons=["%s; the same step runs again under a new id (attempt %d)" % (_short(how, 400), att[key])])

    def _proposal_of(self, rec):
        """A lost run's proposal rebuilt from its execution record (the
        proposal itself was in the state that was not written): the same id,
        lever, action, params (with the price), trigger and experiments."""
        lever = rec["lever"]
        try:
            row = LS.row(lever)
        except LS.LeverError:
            row = {"follow": "job"}              # L4, the label audit (_audit_proposal): a job
        params = dict(rec.get("params") or {}, **(rec.get("meta_params") or {}))
        return {"id": rec["proposal_id"], "lever": lever, "family": LS.family(lever) if row.get("policy_action") else lever,
                "argv": list(rec.get("argv") or []), "params": params, "policy_action": rec.get("action"),
                "risk": rec.get("risk"), "trigger": list(rec.get("trigger") or []), "cites": [], "lit": [],
                "est_gpu_hours": _num(params.get("est_gpu_hours")) or 0.0, "proposed_by": AUTO,
                "lane": LANE_OF.get(lever), "follow": row.get("follow"), "parent_exp": rec.get("parent_exp"),
                "child_exp": rec.get("child_exp"), "attempt": None, "rebuilt_from_run": rec.get("run_id")}

    def _adopt_run(self, ln, it, rec, how):
        """An executed run taken back as lane ln's running item `it`: the lane
        follows it as any item it ran (a lab job by its result, _poll_lab; a
        cluster job by squeue and sacct, an experiment by name, _fold; a login
        verb is done at once, its effect happened), and _on_started's
        bookkeeping, lost with the state, is done again."""
        st = self.st
        p = it["proposal"]
        pay = ((rec.get("remote") or {}).get("payload")) or {}
        pay = pay if isinstance(pay, dict) else {}
        it.update(status="running", job_ids=[str(j) for j in rec.get("job_ids") or []],
                  approval_id=rec.get("approval_id") or it.get("approval_id"), executed_utc=rec.get("ts"),
                  lab_job=self._lab_job_of(rec), adopted_run=rec.get("run_id"), adopted_utc=self.utc)
        lane = st["lanes"][ln]
        lane["item"] = it
        lane["phase"] = self._phase_of(it["lever"])
        st["adopted_runs"] = (list(st.get("adopted_runs") or []) + [rec.get("run_id") or p["id"]])[-ADOPTED_KEEP:]
        self._ledger("adopted_run", lane=ln, lever=it["lever"], proposal_id=p["id"], run_id=rec.get("run_id"),
                     executed_utc=rec.get("ts"), lab_job=it.get("lab_job"), job_ids=it.get("job_ids"),
                     child_exp=p.get("child_exp"), reasons=[_short(how, 500)])
        self._on_started(ln, it)
        if p.get("follow") == "login" or p.get("policy_action") in DONE_AT_ONCE:
            self._done(ln, it, pay)

    def _recover_lab(self, ln, it, why):
        """A lab item refused as already run (its id ran before the state that
        holds the item was written): the lane takes that run back by its
        recorded lab job (_adopt_run), so its result is read and its effect
        recorded, or it fails and is proposed again under a new id. A run a
        lane already followed to its end (_run_followed: a deterministic id
        proposed again for the same work, as an L15 with the same classes was
        before L15 took a new attempt when it finished) is never taken back,
        or its old result would be read as new: the step runs again under a
        new id (_renew). Declined, with a card, only when no lab job of it was
        recorded (the restart came before the hook launched it) or its run was
        taken back once already. Before this, every such item was declined as
        if its effect showed in the snapshots, which an L16S candidate sync's
        never does."""
        st, p = self.st, it["proposal"]
        rec = X._prior_run(self._executions(), p["id"])
        job = self._lab_job_of(rec)
        if job and self._run_followed(rec):
            self._renew(ln, p.get("lever") or it["lever"], p, rec,
                        "refused as already run (%s); that run (lab job %s) was followed to its end, so it is not "
                        "taken back" % (_short(why, 200), job))
            lane = st["lanes"][ln]
            # an L15 is taken out of D29's wait once its date passed: the lane goes back to that wait (no new
            # wait starts) and the renewed discovery is proposed on the next tick
            wait = it["lever"] == "L15" and lane.get("until_utc") and (_secs(lane["until_utc"]) or 0) <= self.now
            self._clear(ln)
            if wait:
                lane["phase"] = "WAIT_DATA"
            return
        if job and (rec.get("run_id") or p["id"]) not in (st.get("adopted_runs") or []):
            return self._adopt_run(ln, it, rec, "refused as already run (%s): its lab job %s is followed"
                                   % (_short(why, 200), job))
        st["declined"] = (list(st.get("declined") or []) + [p["id"]])[-200:]
        self._card("platform", "%s %s already ran and no lab job of it can be followed: not proposed again"
                   % (it["lever"], p["id"][:12]),
                   "%s (proposal %s) was refused: %s. Its earlier run %s, so the lane cannot follow it; the proposal "
                   "is declined and never proposed again. A person checks what that run did (the execution log, the "
                   "lab_jobs directory of the campaign) and, if it did nothing, runs the step by hand."
                   % (it["lever"], p["id"], _short(why, 300),
                      "was taken back once already" if job else
                      "recorded no lab job (%s)" % ((rec or {}).get("status") or "no record")),
                   lever=it["lever"], trigger=list(p.get("trigger") or []))
        self._clear(ln)

    def _adopt_declined(self, ln, lever, p, did, d):
        """True when a declined lab proposal is not skipped: its earlier run
        (an executed run with a recorded lab job) is taken back into idle lane
        ln when no lane followed it and it was not adopted before, and the
        step runs again under a new id (_renew) when a lane followed it to its
        end (a repeat _on_result declined as already run, e.g. an L15 with the
        same classes). This recovers an item _on_result declined before
        _recover_lab existed (live: the L16S 22b59a187f1c00415b9c2d70a0858c3f,
        declined on 2026-10-03 15:36Z) without editing the state."""
        rec = X._prior_run(self._executions(), p["id"])
        if not self._lab_job_of(rec):
            return False
        if self._run_followed(rec):
            self._renew(ln, p.get("lever") or lever, p, rec,
                        "%s was declined as already run; that run (lab job %s) was followed to its end"
                        % (p["id"], self._lab_job_of(rec)))
            return True
        if (rec.get("run_id") or p["id"]) in (self.st.get("adopted_runs") or []):
            return False
        it = {"proposal": p, "lever": p.get("lever") or lever, "status": "proposed", "since_utc": self.utc,
              "attempt": p.get("attempt", 0), "trigger": did, "diagnoses": [d]}
        self._adopt_run(ln, it, rec, "declined earlier although its run (lab job %s) was never followed"
                        % self._lab_job_of(rec))
        return True

    # ---- the one ssh: ready items
    def _ready(self):
        out = []
        ap = AP.state(self.domain, root=self.xctx.approvals_root)
        for ln in SUBMIT_ORDER:
            it = self.st["lanes"][ln].get("item")
            if not it:
                continue
            p = it["proposal"]
            if p["policy_action"] in X.LAB_ACTIONS:
                continue
            if it.get("lever") == "L18" and it.get("status") in ("proposed", "filed") and self._cut_recheck(ln, it, ap):
                continue
            if it.get("status") == "proposed":
                out.append((ln, it, None))
            elif it.get("status") == "filed" and it.get("approval_id"):
                a = ap.get(it["approval_id"]) or {}
                if a.get("status") == "approved" and a.get("execution") is None:
                    why = self._misplaced_review(it)
                    if why:
                        self._refuse_misplaced_review(ln, it, why)
                    else:
                        out.append((ln, it, it["approval_id"]))
                elif a.get("status") == "approved":
                    # run by someone else (a person from the INC page): the
                    # lane follows that run instead of waiting for ever
                    self._executed_elsewhere(ln, it, a)
                elif a.get("status") == "denied":
                    self._ledger("denied", lane=ln, lever=it["lever"], approval_id=it["approval_id"],
                                 decided_by=a.get("decided_by") or "human")
                    self.st["declined"] = (list(self.st.get("declined") or []) + [p["id"]])[-200:]
                    self._person_denied(it, a)
                    self._clear(ln)
                elif a.get("status") == "withdrawn":
                    self._approval_withdrawn(ln, it, a)
                elif self.cfg.get("autonomy") == "envelope" and p.get("risk") == "R3" \
                        and it["lever"] in LS.envelope_levers():
                    out.append((ln, it, None))          # re-checked against the envelope each tick
                elif p["policy_action"] in X.GATED_R2_ACTIONS and self.cfg.get("data_autonomy") == "on":
                    # a gated data lever filed for a person (shadow mode, a
                    # placeholder floor, a limit): re-checked against its gates
                    # each tick once a person has set data_autonomy on, so the
                    # lane does not wait for ever on an item filed before
                    out.append((ln, it, None))
        # a segment's cut and the base v3 build are never submitted while the other's outcome is unknown to it
        keep = []
        for ln, it, appr in out:
            why = self._cut_order(it.get("lever"))
            if why:
                self._once("cut_order:%s" % it.get("lever"), why, "waiting", lane=ln, lever=it.get("lever"),
                           reasons=[why])
                continue
            keep.append((ln, it, appr))
        return keep

    def _cut_recheck(self, ln, it, ap):
        """True when a proposed or filed segment cut (L18) is not submitted
        this tick. It waits while a quarantine is pending or not yet in the
        evidence (_quarantine_unseen), and it is withdrawn when D22, read
        again on this tick's evidence with the TRAIN lane idle (the item holds
        the lane, so this tick's own D22 is silent), no longer calls for L18:
        a cut submitted then meets a queue below M and is refused 'short'
        (2026-10-05 05:15Z: L18 filed at 05:03Z, submitted at 05:15Z after the
        snapshot showed Q 319 against M 1,364). A filed item whose approval
        the ticker cannot close (pending) is kept filed instead (_hold_cut). A
        D22 that cannot be read changes nothing, and neither does a filed item
        whose approval a person denied or ran (_ready follows those)."""
        p = it["proposal"]
        a = (ap.get(it["approval_id"]) or {}) if it.get("status") == "filed" and it.get("approval_id") else {}
        if a.get("status") == "denied" or a.get("execution") is not None:
            return False
        why = self._quarantine_unseen()
        if why:
            self._once("cut_wait:%s" % p["id"], why, "waiting", lane=ln, lever="L18", proposal_id=p["id"],
                       reasons=[why])
            return True
        d = self._cut_still_called()
        if d is None or d.get("unknown") or (d.get("fired") and "L18" in (d.get("levers") or [])):
            return False
        self._withdraw_cut(ln, it, "D22 no longer calls for a cut: %s" % _short(d.get("summary"), 300), a)
        return True

    def _cut_still_called(self):
        """D22 on this tick's evidence with the TRAIN lane idle and unheld
        (only whether the queue still calls for a cut), or None when it cannot
        be read."""
        if not isinstance(self._ctx, dict):
            return None
        ctx = copy.deepcopy(self._ctx)
        lanes = ctx.setdefault("lanes", {})
        lanes["TRAIN"] = dict(lanes.get("TRAIN") or {}, busy=False, hold=None)
        try:
            ev = E.from_snapshot(self._record(self._payload), self.sid, context=ctx, domain=LS.funnel_ref(self.dom))
        except (E.EvidenceError, KeyError, TypeError, ValueError):
            return None
        return DS.by_id(DS.detect(ev, self.dom, self.th, prior=self.prior, only=("D22",))).get("D22")

    def _withdraw_cut(self, ln, it, why, a=None):
        """A proposed or filed L18 withdrawn before submission (as _adopt_fork
        withdraws an old stream's items). Its id goes to failed_ids, not to
        declined: D22 calls for the same cut (k, exp) again once the queue
        refills, and a declined id would never be proposed again. A filed
        item is withdrawn only once its approval is closed (_close_unsubmitted,
        an approved one); a pending approval, or one that could not be closed,
        keeps the item filed and unsubmitted (_hold_cut). `a`: the approval as
        _ready read it this tick."""
        p = it["proposal"]
        aid = it.get("approval_id")
        if aid and (a or {}).get("status") == "pending":
            return self._hold_cut(ln, it, why, "its approval %s is pending, and only a person decides an approval"
                                               % aid)
        closed, cwhy = self._close_unsubmitted(aid, why) if aid else (None, "")
        if aid and not closed:
            return self._hold_cut(ln, it, why, "its approval %s could not be closed (%s)" % (aid, cwhy))
        self._ledger("withdrawn", lane=ln, lever=it.get("lever"), approval_id=aid, proposal_id=p["id"],
                     child_exp=p.get("child_exp"), reasons=[why], approval_closed=closed, close_error=cwhy or None)
        self.st["failed_ids"] = (self._failed_ids() + [p["id"]])[-FAILED_IDS_KEEP:]
        self._clear(ln)

    def _hold_cut(self, ln, it, why, held):
        """A filed L18 that D22 no longer calls for, whose approval the ticker
        cannot close: not submitted, and kept filed rather than withdrawn. A
        pending approval stays on the INC page, where a person can approve it
        and run it (executor.execute_approved); withdrawn, its id in
        failed_ids, that run (a segment cut, R3) would be followed by no lane,
        since _adopt_lost_runs reads the id as ended, and D22 would propose
        another cut beside it. Kept filed, a person's run is followed
        (_executed_elsewhere), an approval approved and not yet run is closed
        and the item withdrawn on the next tick that D22 is still silent
        (_withdraw_cut), and, if D22 calls for the cut again, the item goes on
        as any filed item (run under the envelope, or once a person approves
        it). One waiting event and one card per approval."""
        p = it["proposal"]
        aid = it.get("approval_id")
        if self._once("cut_held:%s" % p["id"], [aid, held], "waiting", lane=ln, lever="L18", proposal_id=p["id"],
                      approval_id=aid, child_exp=p.get("child_exp"),
                      reasons=[why, "not submitted and kept filed, not withdrawn: %s" % held]) is None:
            return
        self._card("escalation", "Filed cut %s not submitted: the queue no longer calls for it (approval %s)"
                   % (p.get("child_exp"), aid),
                   "D22 no longer calls for a segment cut (the queue holds fewer than M eligible images), so the "
                   "filed L18 is not submitted; %s, so the item stays filed and holds TRAIN. Approved and not yet "
                   "run, it is closed unsubmitted and withdrawn while D22 stays silent; run from the INC page, the "
                   "lane follows that run, and the cutter refuses it short while the queue is below M. If the queue "
                   "calls for the cut again, the item goes on as any filed cut (run under the envelope, or once "
                   "approved)." % held,
                   lever="L18", trigger=list(p.get("trigger") or []))

    def _misplaced_review(self, it):
        """Why an approved source review in its cluster form (L16R: sbatch
        run_inc_collect.sh fetch) must not be submitted, or "": its source's
        candidate is placed on the lab, whose provider the cluster's collector
        refuses (not_placed_on_cluster). Such a review was filed before the
        lab form (L16RL) existed (mediatum_1717366: approval
        ap-1790700040-42144765, job 47302914). Fail closed: a source with no
        candidate row this tick (a discovered source the last plan, L15, no
        longer lists) is not submitted either, since its placement cannot be
        read, and an L16R carries no --candidates, so the cluster's collector
        could not load a discovered source's record anyway."""
        if it.get("lever") != "L16R":
            return ""
        src = ((it.get("proposal") or {}).get("params") or {}).get("source")
        cand = self._context_candidate(src) if src else {}
        if not cand:
            return ("L16R (approval %s) fetches source %s on the cluster, and the source has no candidate row in this "
                    "tick's candidates (the last plan, L15, no longer lists it, and the collect config has no known "
                    "item of that id): its provider's placement cannot be read, and the cluster's collector could not "
                    "load its record (an L16R carries no --candidates), so the approved cluster fetch is not "
                    "submitted. If the source is listed again, D20 fetches it (L16 or L16L, by its placement) or files "
                    "its review again in the form its placement calls for" % (it.get("approval_id"), src))
        if cand.get("placement") != "lab":
            return ""
        return ("L16R (approval %s) fetches source %s on the cluster, and its provider %s is placed on the lab "
                "(intake/placement.json, the collect config's lab_only): the cluster's collector refuses it "
                "(not_placed_on_cluster), so the approved cluster fetch is not submitted. The source is fetched on "
                "the lab instead: D20's L16 -> L16L once its pre-check passes, else its review filed again in its lab "
                "form (L16RL) for a person" % (it.get("approval_id"), src, cand.get("provider")))

    def _refuse_misplaced_review(self, ln, it, why):
        """Not submitted (_misplaced_review): recorded, declined, the lane
        cleared, nothing charged and no attempt or failure counted; the
        source's review is superseded so that _file_reviews files it again in
        the form its placement calls for. The approval is closed in the
        approvals log (_close_unsubmitted), so neither the ticker nor a
        person's Run now on the INC page (executor.execute_approved) can still
        send it to sbatch, where the cluster's collector would refuse it."""
        p = it["proposal"]
        src = (p.get("params") or {}).get("source")
        aid = it.get("approval_id")
        closed, cwhy = self._close_unsubmitted(aid, why)
        self._ledger("refused", lane=ln, lever=it["lever"], approval_id=aid, proposal_id=p["id"],
                     source=src, reasons=[why], charged=False, approval_closed=closed,
                     close_error=cwhy or None)
        self.st["declined"] = (list(self.st.get("declined") or []) + [p["id"]])[-200:]
        rv = (self.st.get("source_reviews") or {}).get(src)
        if isinstance(rv, dict):
            rv.update(status="superseded", superseded_utc=self.utc, superseded_reason=_short(why, 500))
        note = ("approval %s is closed (not_submitted, no job), so it cannot be run again" % aid if closed else
                "approval %s could not be closed (%s): do not run it from the INC page, the cluster's collector "
                "refuses it" % (aid, cwhy))
        self._card("data", "Approved review of %s not submitted (approval %s)" % (src, aid), "%s. %s" % (why, note),
                   lever=it["lever"])
        self._clear(ln)

    def _close_unsubmitted(self, aid, why):
        """(closed, why not) for an approval the lane will not submit: its one
        run is recorded as started, then failed with the outcome not_submitted
        and no job, in the approvals log only (approvals.record_executed), so
        approvals.awaiting_execution no longer lists it and
        executor.execute_approved refuses it as already executed. The execution
        log, the budget and the source's attempts are not touched."""
        now = self.xctx.clock()
        got = AP.record_executed(self.domain, aid, "started", AUTO, now, root=self.xctx.approvals_root)
        if not got.get("ok"):
            return False, got.get("reason") or "the approval could not be claimed"
        got = AP.record_executed(self.domain, aid, "failed", AUTO, now, root=self.xctx.approvals_root,
                                 outcome={"status": "not_submitted", "job_ids": [], "error": _short(why, 1000)})
        if not got.get("ok"):
            # claimed, so it cannot run, but its outcome is not recorded
            return False, got.get("reason") or "the approval's outcome could not be written"
        return True, ""

    def _executed_elsewhere(self, ln, it, a):
        """A lane item whose approval another executor ran (a person from the
        INC page): the lane takes that run's outcome."""
        ex = a.get("execution") or {}
        oc = ex.get("outcome") or {}
        if ex.get("phase") == "done" and oc.get("status") == "executed":
            self._ledger("executed_elsewhere", lane=ln, lever=it["lever"], approval_id=it["approval_id"],
                         executed_by=ex.get("executed_by"), job_ids=oc.get("job_ids"))
            self._on_result(ln, it, {"status": "executed", "job_ids": list(oc.get("job_ids") or []),
                                     "approval_id": it["approval_id"], "decided_by": a.get("decided_by"),
                                     "basis": "approval executed by %s" % ex.get("executed_by")})
        elif ex.get("phase") in ("done", "failed"):
            self._failed(ln, it, "approval %s, run by %s, ended %s: %s" % (
                it["approval_id"], ex.get("executed_by"), ex.get("phase"), _short(oc.get("error") or "", 300)),
                charged=True)
        else:
            self._once("elsewhere:%s" % it["approval_id"], ex, "waiting", lane=ln, lever=it["lever"],
                       approval_id=it["approval_id"],
                       reasons=["approval %s is being run by %s (started %s); the lane waits for its outcome"
                                % (it["approval_id"], ex.get("executed_by"), ex.get("started_at"))])

    def _approval_withdrawn(self, ln, it, a):
        """A filed item whose approval its requester withdrew (approvals.withdraw,
        _withdraw_refetch) and that still holds its lane (a tick that withdrew
        it and did not write its state): it leaves the lane as withdrawn, its id
        in failed_ids; never a denial, so no source is closed."""
        p = it["proposal"]
        self._ledger("withdrawn", lane=ln, lever=it.get("lever"), approval_id=it.get("approval_id"),
                     proposal_id=p["id"], withdrawn_by=a.get("withdrawn_by"),
                     reasons=[_short(a.get("withdrawal_reason") or "its approval was withdrawn", 600)])
        self.st["failed_ids"] = (self._failed_ids() + [p["id"]])[-FAILED_IDS_KEEP:]
        self._clear(ln)

    def _person_denied(self, it, a):
        """What a person's denial means beyond the declined proposal: a denied
        leak quarantine (L24 on D28) keeps the source, which lifts D28's TRAIN
        hold for it; a denied source review is recorded against the source."""
        p = it.get("proposal") or {}
        params = p.get("params") or {}
        src = params.get("source")
        if it.get("lever") == "L24" and src and params.get("cite") == "D28":
            self.st["sources"].setdefault(str(src), {})["leak_kept_by"] = a.get("decided_by") or "human"
            self._ledger("leak_kept", source=src, decided_by=a.get("decided_by") or "human",
                         approval_id=it.get("approval_id"))
        elif it.get("lever") == "L24" and src and params.get("cite") == "D31":
            # a denied data-blame quarantine keeps the source: D31 does not call for L24 on it again (its blames
            # are permanent; before 2026-10-26 the same id came back on every tick and stalled lane STOP)
            self.st["sources"].setdefault(str(src), {})["blame_kept_by"] = a.get("decided_by") or "human"
            self._ledger("blame_kept", source=src, decided_by=a.get("decided_by") or "human",
                         approval_id=it.get("approval_id"))
        elif it.get("lever") in DENIED_CLOSES and src and it.get("lever") not in REVIEW_LEVERS:
            # a person denied a step of one source on the DATA lane (a fetch, an intake, a sync, its names): the
            # source is closed with the person's decision as its reason, so D20 goes on with the next source
            # (2026-10-26: a denied L16L stayed the top candidate, its declined id came back on every tick and the
            # lane was carded stalled); stream reopen --source brings it back
            s = self.st["sources"].setdefault(str(src), {})
            if s.get("status") not in ("closed", "quarantined"):
                why = "a person (%s) denied %s %s (approval %s)" % (a.get("decided_by") or "human", it.get("lever"),
                                                                    p.get("id"), it.get("approval_id"))
                s.update(status="closed", closed_reason=_short(why, 300), closed_utc=self.utc,
                         closed_by=a.get("decided_by") or "human")
                self._ledger("source_closed", source=src, lever=it.get("lever"), proposal_id=p.get("id"),
                             decided_by=a.get("decided_by") or "human", approval_id=it.get("approval_id"),
                             reasons=[why])
        if it.get("lever") in REVIEW_LEVERS and src:
            rv = (self.st.get("source_reviews") or {}).get(src)
            if isinstance(rv, dict):
                rv["status"] = "denied"

    # ---- items a person decides, filed without holding their lane
    def _file_person(self, lane_name, lever, pr, did, d):
        """An R3 item only a person may decide (LH, a funnel_F9 release at its
        deadline) is filed without taking its lane, so the lane keeps working
        while it waits; once approved it is adopted into the lane and run."""
        st = self.st
        try:
            p = self._proposal(lever, pr, did, d)
        except (LS.LeverError, ValueError, KeyError) as e:
            self._once("render:%s:%s" % (lever, did), str(e), "not_taken", lever=lever,
                       reasons=["cannot render %s: %s" % (lever, _short(e, 300))])
            return
        if p is None or p["id"] in (st.get("declined") or []) or p["id"] in (st.get("person_items") or {}):
            return
        done_at = (st.get("done_keys") or {}).get(_sha([p.get("lever") or lever, p.get("params") or {}])[:16])
        if done_at and done_at >= str(st.get("last_snapshot_utc") or ""):
            return                                # ran since the evidence was taken
        res = X.submit(p, actor=AUTO, campaign=self.camp, ctx=self.xctx)
        if res.get("status") != "filed" or not res.get("approval_id"):
            self._once("person:%s" % p["id"], res.get("reasons"), "not_taken", lever=p.get("lever") or lever,
                       reasons=res.get("reasons") or [str(res.get("status"))])
            return
        self._park(lane_name, p, did, d, res.get("approval_id"))
        self._ledger("filed", lane=lane_name, lever=p.get("lever") or lever, approval_id=res.get("approval_id"),
                     reasons=res.get("reasons"), proposal_id=p["id"], est_su=res.get("est_su"), parked=True)

    def _park(self, lane_name, p, did, d, approval_id):
        self.st.setdefault("person_items", {})[p["id"]] = {
            "lane": lane_name, "lever": p.get("lever"), "proposal": p, "approval_id": approval_id,
            "trigger": did, "diagnoses": [d] if d else [], "filed_utc": self.utc}

    def _adopt_person_items(self):
        """Parked items a person has decided: an approved one takes its lane
        when the lane is free (and runs through the approval); a denied one is
        declined; one run elsewhere is dropped."""
        st = self.st
        items = st.get("person_items") or {}
        if not items:
            return
        ap = AP.state(self.domain, root=self.xctx.approvals_root)
        for pid in sorted(items, key=lambda k: str(items[k].get("filed_utc"))):
            rec = items[pid]
            a = ap.get(rec.get("approval_id")) or {}
            if a.get("status") == "denied":
                it = {"proposal": rec["proposal"], "lever": rec.get("lever"), "approval_id": rec.get("approval_id")}
                self._ledger("denied", lane=rec.get("lane"), lever=rec.get("lever"), approval_id=rec.get("approval_id"),
                             decided_by=a.get("decided_by") or "human")
                st["declined"] = (list(st.get("declined") or []) + [pid])[-200:]
                self._person_denied(it, a)
                items.pop(pid, None)
            elif a.get("status") == "approved" and a.get("execution") is not None:
                self._ledger("executed_elsewhere", lane=rec.get("lane"), lever=rec.get("lever"),
                             approval_id=rec.get("approval_id"), executed_by=(a.get("execution") or {}).get("executed_by"))
                items.pop(pid, None)
            elif a.get("status") == "approved":
                lane = st["lanes"].get(rec.get("lane")) or {}
                if lane.get("item") is not None or lane.get("hold"):
                    continue
                lane["item"] = {"proposal": rec["proposal"], "lever": rec.get("lever"), "status": "filed",
                                "approval_id": rec.get("approval_id"), "since_utc": self.utc, "attempt": 0,
                                "trigger": rec.get("trigger"), "diagnoses": rec.get("diagnoses") or []}
                lane["phase"] = self._phase_of(rec.get("lever"))
                self._ledger("adopted", lane=rec.get("lane"), lever=rec.get("lever"),
                             approval_id=rec.get("approval_id"), proposal_id=pid,
                             decided_by=a.get("decided_by") or "human")
                items.pop(pid, None)

    def _submit_ready(self):
        """True when this tick's ssh went to submissions."""
        if self.st.get("drift"):
            return False
        ready = self._ready()
        if not ready:
            return False
        before = self.ssh.calls
        own = [x for x in ready if x[1]["proposal"].get("risk") == "R3" or x[2]]
        direct = [x for x in ready if x not in own]

        def run_direct():
            if direct and self.ssh.calls == before and self.ssh.left() and not self._paused_reason():
                res_all = X.submit_many([it["proposal"] for _ln, it, _a in direct], actor=AUTO, campaign=self.camp,
                                        ctx=self.xctx)
                for (ln, it, _a), res in zip(direct, res_all):
                    self._on_result(ln, it, res)
        # 6.2 step 2: stop/cancel first. A ready STOP item (a quarantine, an
        # unblock) goes before any build: an R3 build tried first would take
        # the tick's one ssh and cut from a source about to be quarantined.
        stop_first = any(ln == "STOP" for ln, _it, _a in direct)
        if stop_first:
            run_direct()
        # An R3 item (or an approved one) runs in its own call when its grant
        # or approval lets it run, and is filed without any call otherwise:
        # each is tried in priority order until one uses the tick's ssh.
        for ln, it, appr in own:
            if self.ssh.calls > before or not self.ssh.left():
                break
            if appr:
                res = X.execute_approved(appr, self.camp, self.xctx, invoked_by=AUTO, quiet_repeat=True)
            else:
                res = X.submit(it["proposal"], actor=AUTO, campaign=self.camp, ctx=self.xctx)
            self._on_result(ln, it, res)
        if not stop_first:
            run_direct()
        return self.ssh.calls > before

    def _on_result(self, ln, it, res):
        st = self.st
        s = res.get("status")
        p = it["proposal"]
        payload = ((res.get("remote") or {}).get("payload")) or {}
        if s == "executed":
            it.update(status="running", job_ids=list(res.get("job_ids") or []), approval_id=res.get("approval_id")
                      or it.get("approval_id"), executed_utc=self.utc, lab_job=payload.get("lab_job"))
            self._ledger("executed", lane=ln, lever=it["lever"], action=p["policy_action"], job_ids=res.get("job_ids"),
                         approval_id=res.get("approval_id"), decided_by=res.get("decided_by") or res.get("authorized_as"),
                         basis=res.get("basis"), est_su=res.get("est_su"), proposal_id=p["id"],
                         child_exp=p.get("child_exp"), trigger=p.get("trigger"))
            self._mirror_action(p, res)
            self._on_started(ln, it)
            if p.get("follow") == "login" or p["policy_action"] in DONE_AT_ONCE:
                self._done(ln, it, payload)
            return
        if s == "filed":
            if it.get("status") != "filed" or it.get("approval_id") != res.get("approval_id"):
                it.update(status="filed", approval_id=res.get("approval_id"), filed_utc=self.utc)
                self._ledger("filed", lane=ln, lever=it["lever"], approval_id=res.get("approval_id"),
                             reasons=res.get("reasons"), proposal_id=p["id"], est_su=res.get("est_su"))
            else:
                self._once("filed:%s" % p["id"], res.get("reasons"), "waiting", lane=ln, lever=it["lever"],
                           approval_id=res.get("approval_id"), reasons=res.get("reasons"))
            return
        why = "; ".join(res.get("reasons") or []) or str(s)
        if s == "failed":
            if X.never_ran(res):
                self._once("transport:%s" % p["id"], why, "transport_failed", lane=ln, lever=it["lever"],
                           reasons=res.get("reasons"))
                return
            if payload.get("error_kind") == "busy":
                # inc2.stream's lease is held by another writer (exit 3): a
                # transient, never a failed step; the same proposal runs next tick
                self._once("busy:%s" % p["id"], why, "waiting", lane=ln, lever=it["lever"],
                           reasons=["another writer holds stream.lease: %s" % _short(why, 200)])
                if it.get("status") != "filed":
                    it["status"] = "proposed"
                return
            if payload.get("error_kind") == "qos":
                # S21: a qos rejection is a platform defect, not an attempt of the source
                self._ledger("platform_defect", lane=ln, lever=it["lever"], reasons=res.get("reasons"),
                             error_kind="qos")
                self._card("platform", "sbatch refused on qos: a platform defect", why, trigger=list(p.get("trigger") or []))
                st["lanes"][ln]["hold"] = "platform defect: qos refusal (%s); a person checks the partition" % p["policy_action"]
                st["lanes"][ln]["hold_utc"] = self.utc
                self._clear(ln, keep_phase=False)
                return
            if payload.get("error_kind") == "uncertain" and it["lever"] in RECORD_ONLY_LEVERS:
                # the zoo's submit timed out part-way (L23Z): what it queued runs, and its record is written by the
                # jobs themselves; followed by its job names and its record, as an unknown outcome is
                it.update(status="running", uncertain=True, job_ids=[])
                self._ledger("uncertain", lane=ln, lever=it["lever"], reasons=res.get("reasons"),
                             next="followed by its job names; its record decides")
                self._on_started(ln, it)
                return
            if X.uncertain(res):
                if p.get("follow") in ("build", "experiment") and p.get("child_exp"):
                    it.update(status="running", uncertain=True, job_ids=[])
                    self._ledger("uncertain", lane=ln, lever=it["lever"], child_exp=p.get("child_exp"),
                                 reasons=res.get("reasons"), next="the snapshots decide")
                    self._on_started(ln, it)
                    return
                if it["lever"] in RECORD_ONLY_LEVERS:
                    # record only (L23N): its job has a name of its own (a second is refused while it is queued)
                    # and writes once, so it is followed by that name and its record; never a pause
                    it.update(status="running", uncertain=True, job_ids=[])
                    self._ledger("uncertain", lane=ln, lever=it["lever"], reasons=res.get("reasons"),
                                 next="followed by its job name; its record decides")
                    self._on_started(ln, it)
                    return
                self._ledger("uncertain", lane=ln, lever=it["lever"], reasons=res.get("reasons"), next="paused")
                return self._pause("the outcome of %s (%s) is unknown: it may have run on the cluster; a person checks"
                                   % (it["lever"], p["policy_action"]))
            # a zoo submission whose sbatch failed before anything stayed queued (the submit cancels what it
            # queued): the lane's ordinary failure, proposed again, never the record-only card
            return self._failed(ln, it, why, charged=bool(res.get("charged")),
                                submission=it["lever"] == "L23Z" and payload.get("error_kind") == "submit")
        # refused
        if any(t in why for t in WAIT_REFUSALS) or "the campaign is paused" in why:
            self._once("wait:%s" % p["id"], why, "waiting", lane=ln, lever=it["lever"], reasons=res.get("reasons"))
            return
        if "already ran" in why or "already executed" in why:
            self._ledger("recovered", lane=ln, lever=it["lever"], reasons=res.get("reasons"))
            if p.get("follow") in ("build", "experiment") and p.get("child_exp"):
                it["status"] = "running"          # its experiment is followed by name
                self._on_started(ln, it)
                return
            if it["lever"] in RECORD_ONLY_LEVERS:
                # a record-only job (L23N, L23V, L23E, L23C) whose submission ran before the lab stopped (a restart
                # between the ssh and the state write): followed by its job name and its record as an unknown
                # outcome is (_follow_record_only), so a failure of the job is still a card and r0 says so;
                # declining it here would leave a failed job with no card and the item never proposed again
                it.update(status="running", uncertain=True, job_ids=[])
                self._on_started(ln, it)
                return
            if p.get("follow") == "lab":
                # a lab item (L15, L26, L16L, L16RL, L16S) has a lab job to follow, recorded with its run: the
                # lane takes that run back (its effect, e.g. an L16S's candidate_synced, lives only in this
                # state, never in a snapshot); declined only when no lab job was recorded, with a card
                return self._recover_lab(ln, it, why)
            # a job or verb that already ran has no job id to follow here: its
            # effect shows in the snapshots; following it would fail it as a
            # job that "ended unknown" and count toward the lane's stop-loss
            ran_at = self._done_at(p["id"], ledger=True)
            if ran_at:
                # its item ended done (2026-10-05: D31's L4 was refused as already run and filed as declined, and
                # lane MAINT stalled on it for three weeks): done, never declined, never a stall
                self._done_recalled(ln, it["lever"], p, (p.get("trigger") or [it.get("trigger")])[0], ran_at)
                self._clear(ln)
                return
            st["declined"] = (list(st.get("declined") or []) + [p["id"]])[-200:]
            self._clear(ln)
            return
        if "left in the campaign envelope" in why:
            self._ledger("refused", lane=ln, lever=it["lever"], reasons=res.get("reasons"))
            return self._pause("budget: %s cannot be charged to the campaign envelope (%s)" % (it["lever"], _short(why)))
        self._ledger("refused", lane=ln, lever=it["lever"], reasons=res.get("reasons"), proposal_id=p["id"])
        self.st["declined"] = (list(self.st.get("declined") or []) + [p["id"]])[-200:]
        self._failed(ln, it, why)

    def _mirror_action(self, p, res):
        fn = self.log_action
        if not callable(fn):
            return
        try:
            jobs = list(res.get("job_ids") or [])
            fn("inc_autopilot:%s" % p.get("policy_action"),
               {"ok": res.get("status") == "executed", "jobid": jobs[0] if jobs else None, "job_ids": jobs,
                "campaign": self.name, "lever": p.get("lever"), "approval_id": res.get("approval_id"),
                "decided_by": res.get("decided_by"), "cmd": " ".join(p.get("argv") or []),
                "msg": "; ".join(res.get("reasons") or [])})
        except Exception as e:
            self.log.warning("[inc-stream] action log failed: %s" % e)

    def _on_started(self, ln, it):
        """Bookkeeping when an item starts: a source's attempt, an experiment the
        stream now owns, the R0 progress."""
        st = self.st
        p = it["proposal"]
        lever, params = it["lever"], p.get("params") or {}
        if lever in FETCH_LEVERS:
            s = st["sources"].setdefault(params["source"], {})
            s["attempts"] = int(s.get("attempts") or 0) + 1
            s.update(status="fetching", placement="lab" if lever in LAB_FETCH_LEVERS else "cluster",
                     max_bytes=params.get("max_bytes"))
            # a new fetch record: what an earlier fetch synced, or an intake of it that failed, is not this one's
            s.pop("synced", None)
            s.pop("synced_utc", None)
            s.pop("intake_failed_utc", None)
            s.pop("staging_lost_utc", None)
        child = p.get("child_exp")
        if child and child not in st["exps"] and p.get("follow") in ("build", "experiment"):
            st["exps"].append(child)
            if lever == "L18":
                st["segments"].append({"exp": child, "n": len(st["segments"]) + 1, "done": False,
                                       "committed": False, "built_utc": None})
            elif lever == "L20":
                done = [s for s in st["segments"] if s.get("committed")]
                st["milestones"].append({"exp": child, "n": len(st["milestones"]) + 1, "done": False,
                                         "after_segment": len(done), "pool": None})
        r0 = st["stage"].setdefault("r0", {})
        if lever == "L23B":
            b = next((x for x in (self.dom.get("baselines") or {}).get("items") or [] if x["exp"] == child), None)
            if b:
                r0["baseline_%s" % b["id"]] = "building"
        elif lever == "L23N":
            b = self._native_item(params)
            if b:
                r0["native_%s" % b["id"]] = "running"
        elif lever == "L23V":
            r0["base3"] = "running"
        elif lever == "L23E":
            r0["agnostic"] = "running"
        elif lever == "L23C":
            r0["e2"] = "running"
        elif lever == "L23D":
            r0["e2_attr"] = "running"
        elif lever == "L23Z":
            r0["zoo"] = "running"
        elif lever == "L23F":
            r0["e3_score_%s" % params.get("arm")] = "running"
        elif lever == "L23G":
            r0["e3"] = "running"
        elif lever in V3FAM_LEVERS:
            self._v3fam_started(it)
        elif lever in DS1_LEVERS:
            self._ds1_started(it)
        elif lever == "L25":
            r0["stage_a"] = "building"
        elif lever == "L28":
            r0["stage_c"], r0["stage_c_built"] = "building", True
        elif lever == "LP":
            r0["probe"] = self.utc
        for k in ("pilot",):
            ref = (self.dom.get("stage_a") or {}).get("from_exp")
            if lever == "L25" and ref and ref not in st["exps"]:
                st["exps"].append(ref)
        if lever == "L15":
            # (a lost run taken back before this tick's evidence exists reads the candidates file itself)
            cands = ((self.ev.json(E.CONTEXT) or {}).get("candidates") if self.ev is not None else self._candidates())
            st["discover"]["before"] = sorted(c.get("id") for c in cands or [])

    def _clear(self, ln, keep_phase=False):
        lane = self.st["lanes"][ln]
        pid = ((lane.get("item") or {}).get("proposal") or {}).get("id")
        if pid:
            # every item leaves its lane here: its run is not a lost one (_adopt_lost_runs)
            self.st["ended_ids"] = ([x for x in self.st.get("ended_ids") or [] if x != pid] + [pid])[-ENDED_KEEP:]
        lane["item"] = None
        if not keep_phase:
            lane["phase"] = "IDLE"

    def _done(self, ln, it, payload=None):
        st = self.st
        p = it["proposal"]
        lever, params = it["lever"], p.get("params") or {}
        lane = st["lanes"][ln]
        lane["fails"] = 0
        lane["fail_sources"] = []
        if lever in SOURCE_REFUSAL_LEVERS:
            st["refusal_run"] = []                 # a fetch or an intake that ran: the collector works (_refusal_run)
        (st.get("lost_runs") or {}).pop(step_key(lever, params), None)
        self._ledger("item_done", lane=ln, lever=lever, proposal_id=p["id"], child_exp=p.get("child_exp"))
        # an id that ended done is never declined or stalled on when a diagnosis calls for it again (_done_recalled)
        self._note_done(str(p["id"]), {"utc": self.utc, "lever": lever, "exp": p.get("child_exp")})
        if lever == "L4" and p.get("child_exp"):
            # the label audit's result is read by D31 (context /audits); a frozen experiment is snapshotted again
            # until its record carries the audit (_refreeze_pending)
            self._audits_run()[str(p["child_exp"])] = {"utc": self.utc, "proposal_id": p["id"],
                                                       "audit": (p.get("params") or {}).get("audit")}
        # what a later tick diagnoses before the next snapshot still shows the
        # state before this item: the same work is not proposed again on it
        dk = st.setdefault("done_keys", {})
        dk[_sha([lever, p.get("params") or {}])[:16]] = self.utc
        if len(dk) > 64:
            for k in sorted(dk, key=lambda k: dk[k])[:len(dk) - 64]:
                dk.pop(k, None)
        if lever == "L15":
            # the next discovery runs under a new id even with the same classes (the executor runs an id once; a
            # repeat under this one was refused as already run, and once taken back as if this run were new)
            key, att = step_key(lever, params), st.setdefault("attempts", {})
            att[key] = max(int(att.get(key) or 0), int(p.get("attempt") or 0)) + 1
            before = set(st["discover"].pop("before", []) or [])
            after = set(c.get("id") for c in self._candidates())
            new = len(after - before)
            d = st["discover"]
            d.update(last_utc=self.utc, found_new=new, runs=int(d.get("runs") or 0) + 1,
                     empty_runs=(int(d.get("empty_runs") or 0) + 1) if new == 0 else 0)
            self._check_recall()
        elif lever in FETCH_LEVERS:
            self._fetch_ends()[p["id"]] = self.utc
            s = st["sources"].setdefault(params["source"], {})
            s.update(status="fetched", fetched_utc=self.utc)
            if lever in LAB_FETCH_LEVERS:
                # a lab fetch appends to the lab's own sources.jsonl, which no
                # snapshot folds: its completeness comes from its closing line
                got = self._collect_result(payload)
                self._fetch_completeness(params["source"], got.get("complete"), got.get("remaining"), "lab fetch")
        elif lever == "L16S" and params.get("names"):
            s = st["sources"].setdefault(params["source"], {})
            s["names_synced"] = self.utc
            if s.get("status") == "names_pending":
                # the names layer is on the cluster: the intake runs again (its extracted tree is reused)
                s["status"] = "fetched"
                self._ledger("source_status", source=params["source"], status="fetched", by="stream",
                             reasons=["class names resolved on the lab (L26) and pushed (L16S): intake again"])
        elif lever == "L16S" and params.get("candidate"):
            st["sources"].setdefault(params["source"], {})["candidate_synced"] = LS.inc_path(
                self.paths.cand_rel(params["source"]))
        elif lever == "L16S":
            s = st["sources"].setdefault(params["source"], {})
            s["synced"] = True
            s["synced_utc"] = self.utc
        elif lever == "L16I":
            s = st["sources"].setdefault(params["source"], {})
            s["status"] = "intaken"
            s["intaken_utc"] = self.utc
        elif lever == "L17" and params.get("verb") == "admit":
            b = params.get("intake")
            src = next((k for k, v in st["sources"].items() if v.get("batch") == b), None) or next(
                (k for k, v in st["sources"].items() if b in (v.get("batches") or [])), None)
            if src:
                s = st["sources"][src]
                if s.get("status") not in ("quarantined", "closed"):
                    # a quarantine or a closure taken while the admit ran stands (2026-10-06/07: three D28
                    # quarantines were overwritten here)
                    s["status"] = "admitted"
                s["admit_done_utc"] = self.utc
                ab = s.get("admitted_batches")
                if isinstance(ab, list) or s.get("intake_deferred") is not None:
                    # a source recorded before the list existed: the stream admitted each earlier batch while it
                    # was the latest
                    ab = list(ab) if isinstance(ab, list) else [x for x in s.get("batches") or [] if x != b]
                    s["admitted_batches"] = ab + ([b] if b not in ab else [])
                    # its intake left images deferred, or a shard is committed and not admitted: the next shard
                    self._next_shard(src, s)
                # else no summary of its latest batch has been folded (a state written before shards existed,
                # whose snapshots since lacked the summaries): admitted as before, and the sweep decides on its
                # shards once a snapshot holds that summary, never on its absence
        elif lever == "L17":
            st["stage"]["r0"]["step1_%s" % params.get("verb")] = self.utc
        elif lever == "L19":
            for s in st["segments"]:
                if s["exp"] == params.get("exp"):
                    s["committed"] = True
                    s["committed_utc"] = self.utc
        elif lever == "L21":
            rb = st.setdefault("rollbacks", [])
            q = self._artifact("stream/%s/queue_summary.json" % self.sid) or {}
            pend = [x for x in q.get("rollback_pending") or [] if isinstance(x, dict)
                    and x.get("to_pool") == params.get("to")]
            rb.append({"utc": self.utc, "to": params.get("to"), "milestone": pend[-1].get("milestone") if pend else None})
            self._card("research", "X4: leave-one-source-out after the rollback to %s" % params.get("to"),
                       "the platform bisects the suspect increments (L27); X4 stays for what it cannot separate",
                       lever="X4", trigger=[p.get("trigger", [""])[0] if p.get("trigger") else "D25"])
        elif lever == "L24":
            s = st["sources"].setdefault(params["source"], {})
            s.update(status="quarantined", quarantined_utc=self.utc, cite=params.get("cite"))
            self._card("leak" if params.get("cite") == "D28" else "data",
                       "Source %s quarantined (%s)" % (params["source"], params.get("cite")),
                       "L24 ran on a firing %s; reversible by a person" % params.get("cite"),
                       trigger=[params.get("cite")])
        elif lever == "L23":
            if params.get("verb") == "build":
                st["stage"]["r0"]["splits_built"] = True
        elif lever == "L23N":
            b = self._native_item(params)
            if b:
                st["stage"].setdefault("r0", {})["native_%s" % b["id"]] = "done"
        elif lever in ("L23V", "L23E", "L23C", "L23D", "L23G"):
            st["stage"].setdefault("r0", {})[{"L23V": "base3", "L23E": "agnostic", "L23C": "e2",
                                              "L23D": "e2_attr", "L23G": "e3"}[lever]] = "done"
        elif lever == "L23Z":
            st["stage"].setdefault("r0", {})["zoo"] = "done"
            self._zoo_card(params, p)
        elif lever == "L23F":
            st["stage"].setdefault("r0", {})["e3_score_%s" % params.get("arm")] = "done"
        elif lever in V3FAM_LEVERS:
            self._v3fam_item(lever, params)["state"] = "done"
        elif lever in DS1_LEVERS:
            self._ds1_done(it)
        elif lever == "LV":
            key = p.get("verdict_key") or params.get("exp") or params.get("verb")
            st["stage"]["r0"].setdefault("verdicts", {})[key] = self.utc
        elif lever == "L26":
            s = st["sources"].setdefault(params["source"], {})
            got = self._collect_result(payload)
            if s.get("status") == "names_pending" and (got.get("status") == "partial" or int(got.get("errors") or 0)) \
                    and int(s.get("names_partial") or 0) < NAMES_PARTIAL_MAX:
                # the authority failed on some names: L26 again (a new id), the round not yet counted
                s["names_partial"] = int(s.get("names_partial") or 0) + 1
                self._bump_attempts("L26", params)
                dk.pop(_sha([lever, params])[:16], None)      # not done: its rerun is not 'the same work again'
                self._ledger("names_partial", source=params["source"], errors=got.get("errors"),
                             names=got.get("names"), runs=s["names_partial"])
                self._clear(ln)
                return
            s["names_resolved"] = self.utc
            if s.get("status") == "names_pending":
                s["names_rounds"] = int(s.get("names_rounds") or 0) + 1
                s["names_synced"] = None
        elif lever == "L22":
            st["doublings"] = int(st.get("doublings") or 0) + 1
            st["fork_pending"] = self.utc
        elif lever == "L27":
            rb = st.get("rollbacks") or []
            st["bisected"] = rb[-1]["utc"] if rb else self.utc
        self._clear(ln)

    def _failed(self, ln, it, why, charged=False, retry=True, lost=False, ended=None, submission=False):
        """An item that ended without its effect. `ended` is given only by the
        follow of a build whose job ran and ended without its experiment
        ({"refusal": the provenance's refusal line or None, "states": the
        job's sacct states}); a submission that failed or was refused before
        anything was queued never passes it."""
        st = self.st
        p = it["proposal"]
        lever, params = it["lever"], p.get("params") or {}
        lane = st["lanes"][ln]
        if ended is not None:
            # the build job's estimate (its runs included) is released once sacct settled it: the sacct entry
            # holds what it cost, and no experiment's report spend will ever release it (budget.committed)
            self._build_ended_spend(it, why)
        if lever in FETCH_LEVERS:
            # ended: the byte limits count what it fetched, not its max_bytes
            self._fetch_ends()[p["id"]] = self.utc
        if lost:
            # killed from outside (a restart, an OOM kill, a reboot): not the step's or the source's failure. The
            # lane is freed and the same work is proposed again, up to LOST_RUNS_MAX times in a row for one step;
            # after that it counts as a failure, so a job that is always killed still ends in the stop-loss
            # (2026-10-03: a deploy restart killed the zenodo_15808623 fetch, and counting it as the source's
            # third failed attempt closed the largest source the stream had found)
            key = step_key(lever, params)
            lr = st.setdefault("lost_runs", {})
            lr[key] = int(lr.get(key) or 0) + 1
            if lr[key] < LOST_RUNS_MAX:
                # a new proposal id for the retry (the executor runs an id once), as a failed step gets
                att = st.setdefault("attempts", {})
                att[key] = int(att.get(key) or 0) + 1
                st["failed_ids"] = (self._failed_ids() + [p["id"]])[-FAILED_IDS_KEEP:]
                if lever in FETCH_LEVERS and params.get("source"):
                    src = st.setdefault("sources", {}).setdefault(params["source"], {})
                    if src.get("status") == "fetching":
                        src["status"] = "candidate"      # as a failed fetch leaves it, without a failure
                    # the attempt _on_started counted is not the source's: a 4th attempt pauses the campaign (S15)
                    src["attempts"] = max(0, int(src.get("attempts") or 0) - 1)
                self._ledger("failed", lane=ln, lever=lever, reasons=[_short(why, 1000)], proposal_id=p["id"],
                             charged=charged, fails=int(lane.get("fails") or 0), lost=True, lost_runs=lr[key])
                self._clear(ln)
                return
        if lever in V3FAM_LEVERS and not submission:
            return self._v3fam_failed(ln, it, why, charged, ended=ended)
        if lever in DS1_LEVERS and not submission:
            return self._ds1_failed(ln, it, why, charged, ended=ended)
        if lever in RECORD_ONLY_LEVERS and not submission:
            # record only (a measurement arm's native-resolution rescore; E1's base v3 build and agnostic
            # rescore): a person's card; the lane's failure count, its step count and its stop-loss are not
            # touched, and the item stays failed, so DR0 does not propose it again (a person reruns it, or
            # leaves it)
            r0 = st["stage"].setdefault("r0", {})
            if lever == "L23N":
                b = self._native_item(params)
                if b:
                    r0["native_%s" % b["id"]] = "failed"
                rerun = "inc2.baseline rescore-native --exp %s" % params.get("exp")
            elif lever == "L23V":
                r0["base3"] = "failed"
                rerun = "inc2.base3 build --stream %s (after moving splits/v3 aside if it is partial)" % params.get(
                    "stream")
            elif lever == "L23E":
                r0["agnostic"] = "failed"
                rerun = "inc2.baseline rescore-agnostic --exp %s --reference %s" % (params.get("exp"),
                                                                                   params.get("reference"))
            elif lever == "L23C":
                r0["e2"] = "failed"
                rerun = "inc2.baseline rescore-e2 (sbatch run_inc2_build.sh inc2.baseline rescore-e2)"
            elif lever == "L23D":
                r0["e2_attr"] = "failed"
                rerun = "inc2.baseline rescore-e2-attr (sbatch run_inc2_build.sh inc2.baseline rescore-e2-attr)"
            elif lever == "L23Z":
                r0["zoo"] = "failed"
                rerun = self._zoo_rerun(params)
            elif lever == "L23F":
                r0["e3_score_%s" % params.get("arm")] = "failed"
                rerun = ("inc2.twostage score-arm --arm %s (sbatch run_inc2_build.sh inc2.twostage score-arm --arm %s; "
                         "E3's other arms wait until its record is complete, and E3's verdict (L23G) needs all three "
                         "arms; E2-C's attribution (L23D) never waits for E3)" % (params.get("arm"), params.get("arm")))
            elif lever == "L23G":
                r0["e3"] = "failed"
                rerun = ("inc2.twostage verdict (sbatch run_inc2_build.sh inc2.twostage verdict; it needs all three "
                         "arms' scores; E2-C's attribution (L23D) never waits for E3)")
            else:
                rerun = "the lever's command"
            st["declined"] = (list(st.get("declined") or []) + [p["id"]])[-200:]
            self._ledger("failed", lane=ln, lever=lever, reasons=[_short(why, 1000)], proposal_id=p["id"],
                         charged=charged, record_only=True)
            title = RECORD_ONLY_TITLES[lever]
            self._card("escalation", title % (params.get("exp") or params.get("arm")) if "%s" in title else title,
                       "%s. Record only: the stream runs on and it is not proposed again; a person reruns %s or "
                       "leaves it" % (_short(why, 400), rerun), lever=lever, trigger=list(p.get("trigger") or []))
            self._clear(ln)
            return
        mb = self._measure_build(lever, params, p) if ended is not None else None
        if mb is not None:
            # a measurement arm's build whose job ran and ended without its experiment (MEASURE_BUILD_TITLE):
            # record only, as its runs are; the lane's failure count is not touched, and the next measure item is
            # not stopped by this one's refusal. A job killed from outside (no refusal line, not cancelled) is
            # built again under a new id, up to LOST_RUNS_MAX times in a row (/stage/baselines says missing
            # again, so DR0 proposes it); a refusal of the build itself, a cancelled job, or the last of those
            # retries leaves the item failed (/stage/baselines), and DR0 does not propose it again
            r0 = st["stage"].setdefault("r0", {})
            states = list(ended.get("states") or [])
            refusal = ended.get("refusal")
            cancelled = any(str(x).startswith(CANCELLED_STATES) for x in states)
            key = step_key(lever, params)
            tries = ""
            if not refusal and not cancelled:
                lr = st.setdefault("lost_runs", {})
                lr[key] = int(lr.get(key) or 0) + 1
                if lr[key] < LOST_RUNS_MAX:
                    att = st.setdefault("attempts", {})
                    att[key] = int(att.get(key) or 0) + 1
                    st["failed_ids"] = (self._failed_ids() + [p["id"]])[-FAILED_IDS_KEEP:]
                    r0.pop("baseline_%s" % mb["id"], None)
                    self._ledger("failed", lane=ln, lever=lever, reasons=[_short(why, 1000)], proposal_id=p["id"],
                                 charged=charged, record_only=True, child_exp=mb["exp"], lost=True,
                                 lost_runs=lr[key], states=states, next="built again under a new id")
                    self._clear(ln)
                    return
                tries = " It ended without a refusal line (sacct %s) %d times in a row, so it is not built again." % (
                    ",".join(states) or "unknown", lr[key])
                lr.pop(key, None)
            r0["baseline_%s" % mb["id"]] = "failed"
            st["declined"] = (list(st.get("declined") or []) + [p["id"]])[-200:]
            self._ledger("failed", lane=ln, lever=lever, reasons=[_short(why, 1000)], proposal_id=p["id"],
                         charged=charged, record_only=True, child_exp=mb["exp"], states=states)
            after = ""
            if mb.get("requires") == "e1" and mb.get("e2") == "C":
                after = (" E2-C's attribution (L23D) needs all three of E2-C's experiments, and E2's other builds "
                         "wait while this one is failed (one card): nothing more of E2 is built until a person "
                         "builds %s (inc2.baseline build --exp %s --manifest INC_DIR/%s --seeds %s --arm %s --role %s "
                         "--e2 C; the wait ends once its experiment exists), or leaves E2-C incomplete; E2's verdict "
                         "(L23C) does not wait for it."
                         % (mb["exp"], mb["exp"], mb.get("manifest"), mb.get("seeds"), mb.get("arm"), mb.get("role")))
            elif mb.get("requires") == "e1":
                after = (" E2's verdict (L23C) needs all six of E2's experiments, and E2's other builds wait while "
                         "this one is failed (one card): nothing more of E2 is proposed until a person builds %s "
                         "(inc2.baseline build --exp %s --manifest INC_DIR/%s --seeds %s --arm %s --role %s --e2 %s; "
                         "the wait ends once its experiment exists), or leaves E2 incomplete."
                         % (mb["exp"], mb["exp"], mb.get("manifest"), mb.get("seeds"), mb.get("arm"), mb.get("role"),
                            mb.get("e2")))
            self._card("escalation", MEASURE_BUILD_TITLE % mb["exp"],
                       "%s.%s Record only: the stream runs on and the build is not proposed again.%s"
                       % (_short(why, 600), tries, after), lever=lever, trigger=list(p.get("trigger") or []))
            self._clear(ln)
            return
        if not retry:
            # a refusal the identical request meets again (a verifier or LOCK
            # mismatch, S3): declined for good, its source held, a person decides
            st["declined"] = (list(st.get("declined") or []) + [p["id"]])[-200:]
            src = params.get("source") or next((k for k, v in st["sources"].items()
                                                if v.get("batch") == params.get("intake")), None)
            if src:
                st["sources"].setdefault(src, {}).update(status="held", held_reason=_short(why, 300))
            self._card("research", "X11: the verifier or the LOCK changed under %s" % lever, why, lever="X11",
                       trigger=list(p.get("trigger") or []))
            self._ledger("not_taken", lane=ln, lever=lever, proposal_id=p["id"], retry=False,
                         reasons=["refused with a refusal the identical request meets again: %s" % _short(why, 300)])
            self._clear(ln)
            return
        # one source's repeated failure counts against that source (closed after its 3 failures, below), never twice
        # against the lane: the lane's stop-loss counts consecutive failures of distinct sources (a fetch that fails
        # on every source is the machine's), and D20 ranks a failed source after the others (2026-10-26: a dead
        # link proposed again under a new id held lane DATA on its second failure, as the oversized palmer did)
        src = None
        if lever in FETCH_LEVERS + ("L16I", "L17") and params.get("source") or lever in FETCH_LEVERS:
            src = params.get("source") or next((k for k, v in st["sources"].items()
                                                if v.get("batch") == params.get("intake")), None)
        run = list(lane.get("fail_sources") or []) if int(lane.get("fails") or 0) else []
        repeat = bool(src) and ln == "DATA" and str(src) in run
        if not repeat:
            lane["fails"] = int(lane.get("fails") or 0) + 1
            if src and ln == "DATA":
                run.append(str(src))
        lane["fail_sources"] = run[-8:]
        key = step_key(lever, params)             # the proposal's params: its policy params and price
        att = st.setdefault("attempts", {})
        att[key] = int(att.get(key) or 0) + 1
        st["failed_ids"] = (self._failed_ids() + [p["id"]])[-FAILED_IDS_KEEP:]
        sf = st.setdefault("step_failures", {})
        sf[key] = {"n": int((sf.get(key) or {}).get("n") or 0) + 1, "lever": lever, "lane": ln,
                   "last": _short(why, 300), "utc": self.utc, "proposal_id": p["id"]}
        st["refusals"] = (list(st.get("refusals") or []) + [{"builder": p.get("policy_action"),
                                                              "message": _short(why, 1000)}])[-REFUSALS_KEEP:]
        self._ledger("failed", lane=ln, lever=lever, reasons=[_short(why, 1000)], proposal_id=p["id"],
                     charged=charged, fails=lane["fails"], **({"source_repeat": src} if repeat else {}))
        if src:
            s = st["sources"].setdefault(src, {})
            s["failures"] = int(s.get("failures") or 0) + 1
            # a source part-way through its intake shards whose next shard failed waits for that shard again
            # (D20 never fetches it anew: a new fetch record would start the source over); a hold the
            # collector's ledger brought this tick stays until the collector releases it; a source whose intake
            # failed on a fetch complete on the cluster is intaken again (fetched: DPIPE's L16I under a new id,
            # never a new fetch, _intake_again); others are candidates again. Its failures count per shard (reset
            # when a shard is admitted, _next_shard)
            held = s.get("status") == "held" and s.get("held_by") == "collector"
            # only a source that is not closed by this failure is intaken again (its third closes it: no event says
            # the intake runs again, and the L16I attempts are not bumped)
            again = lever == "L16I" and s["failures"] < 3 and not held and not DS.mid_shards(s) \
                and self._intake_again(src, s, why)
            s["status"] = ("held" if held else "shard_pending" if DS.mid_shards(s) else "fetched" if again
                           else "candidate") if s["failures"] < 3 else "closed"
            if s["status"] == "closed":
                s["closed_reason"] = "3 failed attempts (7.5)"
                s["closed_utc"] = self.utc
                self._ledger("source_closed", source=src, reasons=["3 failed attempts"])
                if not (s.get("admitted_batches") and int(_num(s.get("admitted_target_boxes")) or 0) > 0):
                    # a source whose admitted shards gave target boxes did not end with zero yield
                    self._zero_yield(src, "failed")
        self._clear(ln)
        if lane["fails"] >= int(LS.t(self.th, "stop_loss", "lane_fails")):
            lane["hold"] = "stop-loss: %d consecutive failed steps (last: %s)" % (lane["fails"], _short(why, 200))
            lane["hold_utc"] = self.utc
            self._card("stop_loss", "Lane %s held" % ln, lane["hold"])
            self._ledger("lane_held", lane=ln, hold=lane["hold"])

    def _collector_refusal(self, it, rc, got):
        """The collector's refusal of item `it` when it is the source's own outcome (SOURCE_REFUSAL_*), else None:
        {"code", "detail", "action", "risk", "kind": close | replan | person | retry, "size"}. `got` is the
        collector's closing JSON (a lab process) or the code and detail of its refusal line (a cluster job, whose
        action is unknown: the code's). Only a fetch or an intake of one source that exited 2. A byte cap below the
        plan (the lab's free disk, the collector's byte envelope or its per-source budget, read from any of the
        collector's over_byte_plan forms, _byte_cap_of) is not the source's size, and a detail naming the machine
        (ENVIRONMENT_RE) is not the source's: the lane's ordinary failure."""
        if it.get("lever") not in SOURCE_REFUSAL_LEVERS or str(rc) != "2" or not isinstance(got, dict):
            return None
        params = (it.get("proposal") or {}).get("params") or {}
        code, detail = str(got.get("code") or ""), str(got.get("detail") or got.get("error") or "")
        action, risk = got.get("action"), got.get("risk")
        if not params.get("source") or not code or got.get("status") not in (None, "held", "closed", "refused"):
            return None
        size = None
        if code in SOURCE_REFUSAL_REPLAN:
            cap, size, exact = _byte_cap_of(detail)
            plan = int(_num(params.get("max_bytes")) or 0)
            if cap is not None and plan:
                if exact and cap < 0.99 * plan:
                    return None                # another cap bound it (disk, envelope, budget): not the plan
                if not exact and cap < float("%.2f" % (plan / 1e9)) * 1e9 - 1:
                    return None                # 'no selected file fits in X GB' (X to 0.01 GB) below the plan's X
            kind = "replan"
        elif ENVIRONMENT_RE.search(detail):
            return None                        # the machine's (a missing module, a full disk), not the source's
        elif action == "close" or (action is None and code in SOURCE_REFUSAL_CLOSE):
            kind = "close"
        elif code in SOURCE_REFUSAL_PERSON:
            kind = "person"
        elif code in SOURCE_REFUSAL_RETRY:
            kind = "retry"
        else:
            return None
        return {"code": code, "detail": _short(detail, 400), "action": action, "risk": risk, "kind": kind,
                "size": size}

    def _source_refused(self, ln, it, ref, why):
        """A fetch or an intake the collector refused for a reason of the source's own (_collector_refusal): the
        source's outcome, never the lane's failure. Neither the lane's failure count nor the step's is touched, no
        refusal is recorded against a builder, the run's id is retired and the lane is freed, so the next source
        is proposed (2026-10-07: two identical refusals of one oversized source held the DATA lane on its
        stop-loss for 19 days). The source:
          replan  (over_byte_plan, its attempts not used up) -> a candidate again, planned once at a bounded larger
                  plan (_replan_bytes; a new id); with no larger plan to give, or refused again, closed;
          close   closed with the code and detail, in the zero-yield run (S20);
          person  held for a person's review (one data card; stream reopen --source releases it), in the zero-yield
                  run;
          retry   one failed attempt of the source (closed after its 3, as _failed counts them), ranked after the
                  other candidates.
        REFUSAL_RUN_HOLD sources in a row refused with the same code (close, person or retry) hold the lane: a
        refusal every source meets is no source's own (a licence policy, a normaliser or a listing that changed)."""
        st = self.st
        p = it["proposal"]
        lever, params = it["lever"], p.get("params") or {}
        src = str(params.get("source"))
        lane = st["lanes"][ln]
        if lever in FETCH_LEVERS:
            self._fetch_ends()[p["id"]] = self.utc     # its bytes count from the collector's ledger, not max_bytes
        st["failed_ids"] = (self._failed_ids() + [p["id"]])[-FAILED_IDS_KEEP:]
        s = st["sources"].setdefault(src, {})
        lim = int((LS.limits("L16") or {}).get("attempts_per_source") or 3)
        plan = int(_num(params.get("max_bytes")) or 0)
        kind = ref["kind"]
        reason = "%s: %s" % (ref["code"], ref["detail"])
        new, why_not = None, None
        if kind == "replan":
            if _num(s.get("byte_plan")):
                why_not = "it was planned once already at a larger plan (%.3f GB)" % (_num(s["byte_plan"]) / 1e9)
            elif int(s.get("attempts") or 0) >= lim:
                why_not = "its %d attempts (attempts_per_source %d) are used" % (int(s.get("attempts") or 0), lim)
            else:
                new, why_not = self._replan_bytes(src, plan, ref.get("size"))
            if new is None:
                kind = "close"
        if kind == "replan":
            s.update(status="shard_pending" if DS.mid_shards(s) else "candidate", byte_plan=new,
                     byte_plan_why=_short(reason, 300), byte_plan_utc=self.utc)
            self._ledger("source_replanned", lane=ln, lever=lever, source=src, proposal_id=p["id"],
                         max_bytes=plan, byte_plan=new, size=ref.get("size"), reasons=[_short(reason, 500)],
                         next="proposed once more at %d bytes under a new id" % new)
        elif kind == "close":
            if ref["code"] in SOURCE_REFUSAL_REPLAN:
                reason = ("over_byte_plan: refused at a plan of %.3f GB (%s); not planned again: %s. Collecting it "
                          "with more bytes is a person's decision: the collect config's budgets.approved_bytes[%s], "
                          "then stream reopen --source %s; its next fetch is then filed for that person's approval "
                          "(the executor's attempts_per_source counts its %d earlier fetches)"
                          % (plan / 1e9, ref["detail"], why_not or "refused at its larger plan", src, src,
                             int(s.get("attempts") or 0)))
            if s.get("status") not in ("closed", "quarantined"):
                s.update(status="closed", closed_reason=_short(reason, 500), closed_utc=self.utc)
                self._ledger("source_closed", source=src, lever=lever, proposal_id=p["id"], code=ref["code"],
                             reasons=[_short(reason, 500)], by="collector")
                if not (s.get("admitted_batches") and int(_num(s.get("admitted_target_boxes")) or 0) > 0):
                    self._zero_yield(src, "refused", reasons={ref["code"]: 1})
        elif kind == "person":
            s.update(status="held", held_by="stream", held_reason=_short(reason, 300), held_utc=self.utc)
            self._ledger("source_held", source=src, lever=lever, proposal_id=p["id"], code=ref["code"],
                         reasons=[_short(reason, 500)], by="collector")
            self._card("data", "Source %s held for a person: %s" % (src, ref["code"]),
                       "%s refused %s: %s. The DATA lane goes on with the next source; a person reviews this one "
                       "(the collect config, then stream reopen --name %s --source %s --by human:<email> --why ...)"
                       % (lever, src, _short(reason, 400), self.name, src),
                       lever=lever, trigger=list(p.get("trigger") or []))
            self._zero_yield(src, "held", reasons={ref["code"]: 1})
        else:
            # retry: the source's own failed attempt; the lane's count is not touched
            s["failures"] = int(s.get("failures") or 0) + 1
            if s["failures"] < lim:
                again = lever == "L16I" and not DS.mid_shards(s) and self._intake_again(src, s, reason)
                s["status"] = "shard_pending" if DS.mid_shards(s) else "fetched" if again else "candidate"
            elif s.get("status") not in ("closed", "quarantined"):
                s.update(status="closed", closed_reason=_short("%d failed attempts, the last %s" % (s["failures"],
                                                                                                 reason), 500),
                         closed_utc=self.utc)
                self._ledger("source_closed", source=src, lever=lever, proposal_id=p["id"], code=ref["code"],
                             reasons=[_short(s["closed_reason"], 500)], by="collector")
                if not (s.get("admitted_batches") and int(_num(s.get("admitted_target_boxes")) or 0) > 0):
                    self._zero_yield(src, "failed", reasons={ref["code"]: s["failures"]})
        self._ledger("not_taken", lane=ln, lever=lever, proposal_id=p["id"], source=src, source_refusal=ref["code"],
                     outcome={"replan": "replanned", "close": "closed", "person": "held", "retry": "failed"}[kind],
                     fails=int(lane.get("fails") or 0),
                     reasons=["%s; the collector's refusal is the source's outcome, not the lane's failure"
                              % _short(why, 600)])
        self._clear(ln)
        if kind != "replan":
            self._refusal_run(src, ref["code"])

    def _refusal_run(self, src, code):
        """The run of sources refused in a row with the source's own refusals (state refusal_run; a fetch or an
        intake that ends done, a person's release or resume clears it): REFUSAL_RUN_HOLD sources in a row refused
        with the same code hold the DATA lane (one card)."""
        st = self.st
        run = [r for r in st.get("refusal_run") or [] if r.get("source") != src] + [
            {"source": src, "code": code, "utc": self.utc}]
        st["refusal_run"] = run[-REFUSAL_RUN_HOLD:]
        last = st["refusal_run"]
        if len(last) >= REFUSAL_RUN_HOLD and len({r["code"] for r in last}) == 1:
            lane = st["lanes"]["DATA"]
            if lane.get("hold"):
                return
            lane["hold"] = ("stop-loss: %d sources in a row refused %s (%s): a refusal every source meets is the "
                            "machine's or the config's, not the sources'" % (len(last), code,
                                                                              ", ".join(r["source"] for r in last)))
            lane["hold_utc"] = self.utc
            self._card("stop_loss", "DATA lane held: %d sources in a row refused %s" % (len(last), code),
                       lane["hold"] + "; a person checks the collector (stream release --name %s --by "
                                      "human:<email> once fixed)" % self.name)
            self._ledger("lane_held", lane="DATA", hold=lane["hold"], sources=last)

    def _replan_bytes(self, src, plan, size):
        """(bytes, None) of the replanned fetch of a source refused over_byte_plan at `plan`, or (None, why not):
        the source's stated size plus 10 % when the provider stated it, else REPLAN_MULT times the plan, at least
        REPLAN_FLOOR_BYTES; never over what the per-source limit (_source_room) or the collect envelope
        (_collect_left) leaves. None when that leaves less than the stated size plus 10 % (or, with no stated size,
        less than twice the refused plan or the per-source room, whichever is smaller): a fetch that cannot be
        larger by that much is not tried again."""
        room = self._source_room(src)
        left = self._collect_left()
        need = int(size * 1.1) + 1 if size else min(2 * plan, room)
        want = max(need, REPLAN_MULT * plan, REPLAN_FLOOR_BYTES)
        got = min(want, room) if left is None else min(want, room, left)
        if got < need or got <= plan:
            if left is not None and left < min(want, room):
                return None, ("the collect envelope leaves %.3f GB (collect_gb_envelope, collect_gb_daily; a person "
                              "may grant more)" % (left / 1e9))
            return None, "the per-source limit leaves %.3f GB" % (room / 1e9)
        return int(got), None

    def _source_room(self, src):
        """The bytes a next fetch of source `src` may still plan under L16's per-source limit, counted as the
        executor's byte limit counts them (executor.fetched_bytes over this campaign's fetch runs: what each ended
        attempt fetched, by its machine's collector ledger; its requested max_bytes when that is unknown)."""
        cap = int(float((LS.limits("L16") or {}).get("gb_per_source") or 50) * 1e9)
        try:
            got = sum(x[2] for x in X.fetched_bytes(self.camp, self._fetch_recs(), self.now) if x[0] == src)
        except Exception:                        # unknown is never room
            return 0
        return max(0, int(cap - got))

    def _collect_left(self):
        """The bytes the campaign's collect envelope (collect_gb_envelope; collect_gb_daily and L16's gb_per_day
        when declared) leaves a next fetch, counted as executor.stream_limits counts them; None when none is
        declared; 0 when the count cannot be taken."""
        c = self.camp
        lim = LS.limits("L16") or {}
        daily = [float(x) for x in (lim.get("gb_per_day"), c.get("collect_gb_daily")) if x is not None]
        if c.get("collect_gb_envelope") is None and not daily:
            return None
        try:
            rows = X.fetched_bytes(c, self._fetch_recs(), self.now)
        except Exception:                        # unknown is never room
            return 0
        caps = []
        if c.get("collect_gb_envelope") is not None:
            caps.append(float(c["collect_gb_envelope"]) * 1e9 - sum(x[2] for x in rows))
        if daily:
            caps.append(min(daily) * 1e9 - sum(x[2] for x in rows if x[1] >= self.now - 86400.0))
        return max(0, int(min(caps)))

    def _fetch_recs(self):
        """This campaign's fetch runs the executor's byte limits count (executed, or charged)."""
        return [r for r in self._executions() if r.get("action") in X.FETCH_ACTIONS
                and (r.get("status") == "executed" or r.get("charged"))]

    def _zero_yield(self, src, how, boxes=None, reasons=None):
        """The DATA lane's zero-yield stop-loss (6.6, S20): 3 consecutive sources
        ending failed, refused, held for a person or with zero admitted target
        boxes hold the lane, with a card listing each source's decision reasons
        (its intake's, and the collector's refusal code when one ended it)."""
        st = self.st
        if how == "yield" and boxes:
            st["zero_run"] = []
            return
        why = self._decision_reasons(src)
        for k, v in (reasons or {}).items():
            why[k] = why.get(k, 0) + int(v or 0)
        run = [r for r in st.get("zero_run") or [] if r.get("source") != src]
        run.append({"source": src, "how": how, "reasons": why, "utc": self.utc})
        st["zero_run"] = run
        n = int(LS.t(self.th, "stop_loss", "zero_yield_sources"))
        if len(run) >= n:
            lane = st["lanes"]["DATA"]
            lane["hold"] = "stop-loss: %d consecutive sources ended failed or with zero admitted target boxes" % len(run)
            lane["hold_utc"] = self.utc
            self._card("stop_loss", "DATA lane held: %d zero-yield sources" % len(run), "; ".join(
                "%s (%s): %s" % (r["source"], r["how"], ", ".join("%s %s" % (k, v) for k, v in
                                                                  sorted((r.get("reasons") or {}).items())) or "no reasons recorded")
                for r in run))
            self._ledger("lane_held", lane="DATA", hold=lane["hold"], sources=run)

    def _fetch_completeness(self, src, complete, remaining=None, basis=""):
        """A source whose last fetch stopped at a byte cap (collect.fetch:
        complete false, the files left recorded as remaining) is `partial`:
        once its admitted shard is observed, D20 may collect its next shard
        (the collector continues from what staging holds), within L16's
        attempt and byte limits. A complete fetch clears it."""
        if not isinstance(complete, bool):
            return
        s = self.st["sources"].setdefault(str(src), {})
        was = bool(s.get("partial"))
        s["partial"] = not complete
        if remaining is not None:
            s["fetch_remaining"] = remaining
        if s["partial"] != was:
            self._ledger("source_partial" if s["partial"] else "source_complete", source=src,
                         remaining=remaining, basis=basis)

    def _intake_failed_after_fetch(self, src, s):
        """The utc of the source's last failed intake (L16I) after its last
        fetch, else None: intake_failed_utc (_intake_again), or, for a failure
        recorded before that field existed (the lab's zenodo_10762138 of
        2026-10-28), the L16I step failure of the source (step_failures)."""
        t = s.get("intake_failed_utc")
        if not t:
            key = step_key("L16I", LS.policy_params("L16I", {"source": src}))
            t = ((self.st.get("step_failures") or {}).get(key) or {}).get("utc")
        return t if t and str(t) >= str(s.get("fetched_utc") or "") else None

    def _intake_again(self, src, s, why):
        """Called when an intake (L16I) of `src` failed: True when the source is
        intaken again on the fetch it has (DS.fetch_on_cluster: complete and on
        the cluster), so it is 'fetched' and DPIPE proposes L16I under a new
        id, never a new fetch; its failures still count (closed after 3). A
        failure that says the staging is gone or changed (STAGING_GONE_RE:
        the collector's not_fetched or StaleInput) takes the fetch off the
        cluster (staging_lost_utc) and returns False: the source is a candidate
        again and D20 may fetch it anew, as before."""
        if STAGING_GONE_RE.search(str(why or "")):
            s["staging_lost_utc"] = self.utc
            return False
        if not DS.fetch_on_cluster(s):
            return False
        s["intake_failed_utc"] = self.utc
        self._ledger("source_status", source=src, status="fetched", by="stream",
                     failures=int(s.get("failures") or 0),
                     reasons=["its intake failed (%s); its fetch is complete and on the cluster (fetched %s, "
                              "fetch_remaining %s%s): the intake runs again (L16I under a new id), never a new "
                              "fetch; closed after 3 failures" % (_short(why, 300), s.get("fetched_utc"),
                                                                  s.get("fetch_remaining"),
                                                                  ", synced" if s.get("placement") == "lab" else "")])
        self._bump_attempts("L16I", {"source": src})
        return True

    def _intake_retry_sweep(self):
        """Every tick, before the diagnoses: a source made a candidate after a
        failed intake (L16I) although its fetch is complete and on the cluster
        (the lab's zenodo_10762138 of 2026-10-28, recorded before _intake_again
        existed) is made 'fetched' again, so DPIPE intakes it, and a fetch of
        it the DATA lane holds (proposed, or filed for a person) is withdrawn
        with its reason (_withdraw_refetch): no source is closed, no card is
        written, nothing waits on a person."""
        st = self.st
        lim = int((LS.limits("L16") or {}).get("attempts_per_source") or 3)
        for src in sorted(st.get("sources") or {}):
            s = st["sources"][src]
            if not isinstance(s, dict) or s.get("status") != "candidate" or int(s.get("failures") or 0) >= lim:
                continue
            if DS.mid_shards(s) or not DS.fetch_on_cluster(s):
                continue
            t = self._intake_failed_after_fetch(src, s)
            if not t:
                continue
            s["status"] = "fetched"
            s["intake_failed_utc"] = t
            self._bump_attempts("L16I", {"source": src})
            self._ledger("source_status", source=src, status="fetched", by="stream", failures=int(s.get("failures")
                                                                                                 or 0),
                         reasons=["made a candidate after its intake failed (%s), although its fetch is complete and "
                                  "on the cluster (fetched %s, fetch_remaining %s%s): the intake runs again (L16I "
                                  "under a new id), never a new fetch" % (t, s.get("fetched_utc"),
                                                                          s.get("fetch_remaining"),
                                                                          ", synced" if s.get("placement") == "lab"
                                                                          else "")])
        lane = st["lanes"]["DATA"]
        it = lane.get("item") or {}
        src = ((it.get("proposal") or {}).get("params") or {}).get("source")
        if it.get("lever") in FETCH_LEVERS and src and it.get("status") in ("proposed", "filed") \
                and (st["sources"].get(src) or {}).get("status") == "fetched" \
                and DS.fetch_on_cluster(st["sources"][src]):
            self._withdraw_refetch("DATA", it, src)

    def _withdraw_refetch(self, ln, it, src):
        """A fetch of `src` (L16, L16L, L16R or L16RL) proposed or filed while
        the source's fetch is complete on the cluster: withdrawn. A pending
        approval is withdrawn by its requester (approvals.withdraw: not a
        denial, so the source is not closed, DENIED_CLOSES); an approved one
        not yet run is closed unsubmitted (_close_unsubmitted); one a person
        is running or ran is followed as any run (the lane keeps it). The id
        goes to failed_ids, the lane is freed; no card."""
        p = it["proposal"]
        aid = it.get("approval_id")
        s = self.st["sources"].get(src) or {}
        why = ("withdrawn: %s's fetch is complete and on the cluster (fetched %s, fetch_remaining %s%s) and its "
               "intake failed (%s); the intake runs again (L16I), and a second fetch of %s would only stage the "
               "same files again" % (src, s.get("fetched_utc"), s.get("fetch_remaining"),
                                     ", synced" if s.get("placement") == "lab" else "",
                                     s.get("intake_failed_utc"), src))
        closed, cwhy = None, ""
        if aid:
            a = AP.state(self.domain, root=self.xctx.approvals_root).get(aid) or {}
            if a.get("status") == "pending":
                got = AP.withdraw(self.domain, aid, AUTO, why, self.xctx.clock(), root=self.xctx.approvals_root)
                closed, cwhy = bool(got.get("ok")), (got.get("reason") or "")
            elif a.get("status") == "approved" and a.get("execution") is None:
                closed, cwhy = self._close_unsubmitted(aid, why)
            elif a.get("status") == "withdrawn":
                closed = True
            else:
                # denied (the lane's own path closes the source as a person decided), or a run a person started:
                # left to the lane's own handling
                return
            if not closed:
                self._once("refetch_kept:%s" % p["id"], [aid, cwhy], "waiting", lane=ln, lever=it.get("lever"),
                           proposal_id=p["id"], approval_id=aid,
                           reasons=[why, "kept filed: its approval %s could not be withdrawn (%s)" % (aid, cwhy)])
                return
        self._ledger("withdrawn", lane=ln, lever=it.get("lever"), approval_id=aid, proposal_id=p["id"], source=src,
                     reasons=[why], approval_withdrawn=closed, withdraw_error=cwhy or None)
        self.st["failed_ids"] = (self._failed_ids() + [p["id"]])[-FAILED_IDS_KEEP:]
        self._clear(ln)

    def _names_pending(self, src, s, row):
        """The collector held `src` for class names the offline resolver lacks
        (an intake's names_pending): the source enters the names round (DPIPE:
        L26, then L16S of the names layer, then L16I again). After
        NAMES_ROUNDS_MAX rounds it is held for a person with a card."""
        names = [str(n) for n in (row.get("pending_names") or [])][:200]
        s["names_pending_ts"] = row.get("pending_ts")
        s["pending_names"] = names
        if int(s.get("names_rounds") or 0) >= NAMES_ROUNDS_MAX:
            why = ("%d class name(s) still unresolved after %d L26 round(s): %s"
                   % (len(names), int(s.get("names_rounds") or 0), ", ".join(names[:10])))
            s.update(status="held", held_by="collector", held_reason=_short(why, 300), names_held=True)
            self._ledger("source_status", source=src, status="held", by="collector", reasons=[_short(why, 300)])
            self._card("data", "Source %s: class names unresolved after L26" % src,
                       "%s; a card table (collect config card_class_tables) or a person's names decision "
                       "maps them" % why, trigger=["DPIPE"])
            return
        s.update(status="names_pending", held_by=None, held_reason=None, names_resolved=None, names_synced=None,
                 names_held=None, names_partial=0)
        # the round's L26 and `L16S --names` have the params of the source's earlier ones (a pre-fetch L26, an
        # earlier round): a fresh attempt count gives them new ids, or the executor would refuse them as run
        self._bump_attempts("L26", {"source": src, "out": str(self.paths.names_dir()) + "/"})
        self._bump_attempts("L16S", {"source": src, "names": 1})
        self._ledger("source_status", source=src, status="names_pending", by="collector",
                     reasons=["the intake's class map lacks %d name(s) (L26 resolves them): %s"
                              % (len(names), _short(", ".join(names[:20]), 300))])

    def _bump_attempts(self, lever, params):
        key = step_key(lever, LS.policy_params(lever, params))
        att = self.st.setdefault("attempts", {})
        att[key] = int(att.get(key) or 0) + 1

    def _next_shard(self, src, s):
        """An admitted source whose intake left images deferred (collect.intake
        step 1b: its latest batch's deferred_remaining), or with a committed
        shard not yet admitted, waits for its next shard (shard_pending): DPIPE
        admits the shard, else proposes L16I again once shard_waits allows,
        and D20 and D21 leave it alone. Each time, the L16I step's attempt
        count rises, so the next intake runs under an id of its own (the
        executor runs an id once), and the source's failure count restarts
        (failures count per shard). A shard that left as many images deferred
        as the shard before it made no progress: the source is closed with a
        card, never proposed again in a loop (a person's reopening resumes
        it). A source whose intake made no new batch (shards_done) is
        complete."""
        if s.get("status") != "admitted":
            return
        left = int(_num(s.get("intake_deferred")) or 0)
        if s.get("shards_done") and s.get("shards_done") == s.get("batch"):
            left = 0
        ab = set(s.get("admitted_batches") or [])
        unadmitted = [b for b in s.get("batches") or [] if b not in ab]
        if left <= 0 and not unadmitted:
            return
        rec = (s.get("shard_log") or {}).get(s.get("batch")) or {}
        prev = (s.get("shard_log") or {}).get(rec.get("prev")) if rec.get("prev") else None
        if left > 0 and not unadmitted and prev and int(rec.get("deferred") or 0) >= int(prev.get("deferred") or 0):
            self._shard_stalled(src, s, "its shard %s left %d image(s) deferred, as many as %s before it"
                                % (s.get("batch"), left, rec.get("prev")))
            return
        s["status"] = "shard_pending"
        s["failures"] = 0
        if left > 0:
            self._bump_attempts("L16I", {"source": src})
        self._ledger("source_status", source=src, status="shard_pending", by="stream", batch=s.get("batch"),
                     deferred=left, unadmitted=unadmitted,
                     reasons=["%s admitted; %d image(s) its intake deferred remain%s: the next shard (L16I, then L17 "
                              "admit), never a new fetch" % (s.get("batch"), left, "; %d committed shard(s) not "
                                                             "admitted yet" % len(unadmitted) if unadmitted else "")])

    def _shard_stalled(self, src, s, why):
        """A source whose intake shards stopped making progress: closed with a
        card for a person (fail closed: the same shard is never proposed again
        and again); `stream reopen` resumes its shards."""
        s.update(status="closed", closed_reason="intake shards made no progress", closed_utc=self.utc)
        self._ledger("source_closed", source=src, reasons=[_short(why, 300)])
        self._card("data", "Source %s: its intake shards made no progress" % src,
                   "%s. Closed so the shard is not proposed again in a loop; the collector's intake summary and "
                   "decisions say why; `stream reopen` resumes the shards" % why, trigger=["DPIPE"])

    def _shard_sweep(self):
        """Once per snapshot fold, after the intake summaries and the
        collector's ledger, deciding only on what this snapshot shows (a
        snapshot whose summaries failed to read decides nothing): (a) a
        source the stream admitted before shards existed, whose latest batch's
        summary this snapshot holds, gets its admitted batches (each of its
        batches, as the stream admitted the latest every time) and, when its
        intake left images deferred, waits for its next shard; (b) a source
        whose continuation intake ended at an earlier snapshot while the
        collector's ledger in this snapshot lists no batch the stream has not
        admitted: complete when the collector's last batch left nothing
        deferred (shards_done), else its shards made no progress (closed with
        a card, never proposed again in a loop)."""
        arts = getattr(self, "_arts", None) or {}
        col = arts.get("intake/sources.json") if isinstance(arts.get("intake/sources.json"), dict) else {}
        for src, s in sorted((self.st.get("sources") or {}).items()):
            if not isinstance(s, dict):
                continue
            seen = isinstance(arts.get("intake/%s/summary.json" % s.get("batch")), dict)
            if s.get("status") == "admitted" and not isinstance(s.get("admitted_batches"), list) and s.get("batch"):
                if not seen or s.get("intake_deferred") is None:
                    continue
                s["admitted_batches"] = list(s.get("batches") or [s["batch"]])
                self._next_shard(src, s)
            elif s.get("status") == "intaken" and s.get("admitted_batches") and s.get("intaken_utc") \
                    and str(s["intaken_utc"]) < self.utc and seen and isinstance(col.get(src), dict) \
                    and all(b in s["admitted_batches"] for b in s.get("batches") or []) \
                    and all(str(b) in (s.get("batches") or []) for b in col[src].get("batches") or []):
                left = int(_num((col[src].get("yield") or {}).get("images_deferred")) or 0)
                if left > 0:
                    self._shard_stalled(src, s, "its intake of the next shard ended without a new batch while the "
                                        "collector's ledger shows %d image(s) deferred" % left)
                    continue
                s.update(status="admitted", shards_done=s.get("batch"), intake_deferred=0)
                self._ledger("source_status", source=src, status="admitted", by="stream", batch=s.get("batch"),
                             reasons=["the intake of the next shard made no new batch and the collector's ledger "
                                      "shows nothing deferred: nothing of the source is left to take"])

    def _intake_names_pending(self, ln, it, ids, states):
        """An L16I job that ended on names_pending (the collector's ledger,
        folded this tick, holds the source for its class names): neither a
        failure of the source nor a step toward the lane's stop-loss. The lane
        is freed and the intake's id retired, so the intake after the names
        round runs under a new id; a source past its names rounds stays held
        for a person."""
        st, p = self.st, it["proposal"]
        params = p.get("params") or {}
        key = step_key(it["lever"], params)
        att = st.setdefault("attempts", {})
        att[key] = int(att.get(key) or 0) + 1
        st["failed_ids"] = (self._failed_ids() + [p["id"]])[-FAILED_IDS_KEEP:]
        src = st["sources"].setdefault(params.get("source"), {})
        if src.get("status") not in ("names_pending", "held"):
            # the job's refusal line says names_pending before the collector's ledger was folded (read before
            # the job ended, or not shipped): the source waits in the round for the fold, whose new hold time
            # brings its names (DPIPE proposes nothing for a round without names)
            src.update(status="names_pending", pending_names=[], names_resolved=None, names_synced=None)
        self._ledger("intake_names_pending", lane=ln, lever=it["lever"], proposal_id=p["id"],
                     source=params.get("source"), job_ids=list(ids), states=list(states),
                     names=list(src.get("pending_names") or [])[:50],
                     reasons=["the intake's class names are not in the offline resolver: L26 first, not a failure"])
        self._clear(ln)

    def _write_pending_names(self, src):
        """The class names a names round resolves (the cluster intake's held
        event, folded by the snapshot) as the lab's
        intake/work/<source>/pending_names.json, which `collect names` reads
        (collect.names.source_names): the cluster's own file never reaches the
        lab."""
        s = (self.st.get("sources") or {}).get(src) or {}
        names = s.get("pending_names") or []
        if s.get("status") != "names_pending" or not names:
            return None
        from ..collect import safe_name
        path = self.paths.lab_inc / "intake" / "work" / safe_name(src) / "pending_names.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        from ..collect import FORMATS as CF
        _write_json(path, {"format": CF["pending_names"], "source": src, "names": list(names),
                           "from": "the cluster intake's held event (names_pending, %s), via the snapshot"
                                   % s.get("names_pending_ts"), "written_utc": self.utc})
        return str(path)

    @staticmethod
    def _collect_result(res):
        """The collector's closing JSON ({...} after '[collect] <verb>: ') in a
        lab process's stdout tail, or {}."""
        for line in reversed(str((res or {}).get("tail") or "").splitlines()):
            m = re.match(r"^\[collect\] [a-z]+: (\{.*\})\s*$", line.strip())
            if m:
                try:
                    out = json.loads(m.group(1))
                    return out if isinstance(out, dict) else {}
                except ValueError:
                    return {}
        return {}

    def _decision_reasons(self, src):
        """{reason: count} of a source's intake decisions (its batches' summaries,
        from the snapshot being folded when there is one)."""
        out = {}
        arts = getattr(self, "_arts", None)
        if arts is None:
            arts = {n: self.ev.json(n) for n in (self.ev.artifacts if self.ev is not None else [])}
        for name in sorted(arts):
            if name.startswith("intake/") and name.endswith("/summary.json"):
                s = arts.get(name) or {}
                if s.get("source") != src:
                    continue
                # collect.intake's summary: its decisions by reason (the guard's and the
                # normaliser's rejections), else the yield's rejected counts
                dec = dict(((s.get("decisions") or {}).get("by_reason")) or {})
                dec.pop("kept", None)
                for k, v in (dec or (s.get("yield") or {}).get("rejected") or s.get("zero_yield_reasons") or {}).items():
                    out[k] = out.get(k, 0) + int(v or 0)
        return out

    def _check_recall(self):
        """S1b: discovery's recall of the owner's known items (collect config);
        a miss is a discovery defect and raises a card; the item stays fetchable.
        The collector's own recall (candidates.json `recall`: a known item is
        found when a search result matched its match rules) is read when
        present; else a known item counts as found when a searched row names it."""
        raw = _read_json(self.paths.candidates())
        rec_c = raw.get("recall") if isinstance(raw, dict) and isinstance(raw.get("recall"), dict) else None
        _p, cc = self._collect_config()
        known = [str(r.get("id") or r.get("source")) for r in (cc or {}).get("known_items") or []
                 if isinstance(r, dict) and r.get("recall", True) is not False]
        if rec_c is not None:
            missed = sorted(str(m.get("id")) for m in rec_c.get("missed") or [] if isinstance(m, dict))
            n = int(rec_c.get("known_items") or len(known))
            basis = "collector"
        else:
            rows = raw.get("candidates") if isinstance(raw, dict) else raw
            found = {str(r.get("source_id") or r.get("id") or r.get("source") or "") for r in rows or []
                     if isinstance(r, dict) and not r.get("known_item")}
            missed = sorted(k for k in known if k not in found)
            n = len(known)
            basis = "rows"
        rec = {"known": n, "found": n - len(missed), "recall": (n - len(missed)) / float(n) if n else None,
               "missed": missed, "basis": basis}
        self.st["discover"]["recall"] = rec
        self._ledger("discovery_recall", **rec)
        if missed:
            self._card("discovery", "Discovery missed %d known item(s)" % len(missed),
                       "a discovery defect (S1b): %s; each stays fetchable by L16" % ", ".join(missed))

    # ---- observe: the one snapshot
    def _r0_exps(self):
        """The rollout's experiments the stream reads whoever built them: the
        baselines, Stage A and the pilot it is judged against, Stage C."""
        dom = self.dom or {}
        out = [b["exp"] for b in (dom.get("baselines") or {}).get("items") or []]
        sa = dom.get("stage_a") or {}
        out += [x for x in (sa.get("exp"), sa.get("from_exp")) if x]
        out.append("%s_c001" % self.sid)
        return out

    def _live_exps(self):
        st = self.st
        names = list(dict.fromkeys(list(st.get("exps") or []) + self._r0_exps()))
        return [e for e in names if e not in (st.get("frozen") or {})] + self._refreeze_pending()

    def _audits_run(self):
        """{exp: {utc, proposal_id, audit}} of the label audits (L4) whose item ended done (state audits_run,
        written by _done). A state written before the record existed takes them once from the campaign ledger's
        item_done lines of L4 (2026-10-05: segment s001's audit of the first fork, job 47429146)."""
        st = self.st
        if not isinstance(st.get("audits_run"), dict):
            if self._ledger_ended() is None:
                return {}                        # read again on the next tick
            got = {}
            for pid, rec in sorted((self._ended_done or {}).items(), key=lambda kv: kv[1].get("utc") or ""):
                if rec.get("lever") == "L4" and rec.get("exp"):
                    got[str(rec["exp"])] = {"utc": rec.get("utc"), "proposal_id": pid, "audit": None}
            st["audits_run"] = got
        return st["audits_run"]

    def _refreeze_pending(self):
        """The frozen experiments snapshotted again so that their record carries the label audit run on them after
        they were frozen (an L4 done after the freeze): the audit is written into the experiment's own directory,
        which the frozen record never saw (2026-10-05: the first fork's segment s001 frozen at 04:23:46Z, its audit
        written at 05:01:53Z, so D31 never read it and called for L4 on every tick). Each is followed in
        st.refreeze until a snapshot of it carries <exp>/audit/label_audit.json (the record is then replaced, the old
        one moved aside, _settle_refreeze), or for BUILD_LOST_SNAPSHOTS snapshots without it (one card)."""
        st = self.st
        rf = st.setdefault("refreeze", {})
        frozen = st.get("frozen") or {}
        for exp, a in sorted(self._audits_run().items()):
            if exp in rf or exp not in frozen:
                continue
            fz = (frozen.get(exp) or {}).get("utc")
            if fz and a.get("utc") and str(a["utc"]) <= str(fz):
                continue                     # frozen after the audit ended: its record was taken with it
            rf[exp] = {"since_utc": self.utc, "audit_utc": a.get("utc"), "proposal_id": a.get("proposal_id"),
                       "snapshots": 0, "state": "pending"}
            self._ledger("refreeze_wanted", exp=exp, audit_utc=a.get("utc"), proposal_id=a.get("proposal_id"),
                         frozen_utc=fz, reasons=["the label audit (L4) on %s ended at %s, after the experiment was "
                                                 "frozen at %s: it is snapshotted again until its record carries "
                                                 "the audit" % (exp, a.get("utc"), fz)])
        return sorted(e for e, r in rf.items() if isinstance(r, dict) and r.get("state") == "pending")

    def _settle_refreeze(self, payload):
        """Each pending refreeze (_refreeze_pending) against this snapshot: a snapshot of the experiment that carries
        its label audit replaces the frozen record (the old one moved aside to frozen/superseded/, never deleted);
        one without it counts, and after BUILD_LOST_SNAPSHOTS such snapshots the refreeze ends with one card."""
        st = self.st
        rf = st.get("refreeze") or {}
        for exp in sorted(e for e, r in rf.items() if isinstance(r, dict) and r.get("state") == "pending"):
            rec = rf[exp]
            snap = (((payload.get("experiments") or {}).get(exp) or {}).get("snapshot")) or {}
            arts = ((snap.get("decision") or {}).get("artifacts")) or {}
            if snap.get("built") and isinstance(arts.get(AUDIT_ARTIFACT % exp), dict):
                path = self.paths.frozen(self.name, exp)
                old = _read_json(path) if path.is_file() else None
                aside = None
                if path.is_file():
                    stamp = re.sub(r"[^0-9A-Za-z]", "", str((old or {}).get("frozen_utc") or self.utc))
                    aside = path.parent / "superseded" / ("%s.%s.json" % (exp, stamp))
                    aside.parent.mkdir(parents=True, exist_ok=True)
                    n = 1
                    while aside.exists():
                        aside = path.parent / "superseded" / ("%s.%s.%d.json" % (exp, stamp, n))
                        n += 1
                    os.replace(str(path), str(aside))
                _write_json(path, {"exp": exp, "frozen_utc": self.utc, "refrozen_from": (old or {}).get("frozen_utc"),
                                   "why": "the label audit (L4) ended at %s, after the first freeze" % rec.get(
                                       "audit_utc"),
                                   "snapshot": {k: v for k, v in snap.items() if k != "display_only"}})
                st.setdefault("frozen", {})[exp] = {"utc": self.utc, "refrozen_from": (old or {}).get("frozen_utc")}
                rec.update(state="done", refrozen_utc=self.utc, moved_aside=str(aside) if aside else None)
                self._ledger("refrozen", exp=exp, audit_utc=rec.get("audit_utc"), moved_aside=rec["moved_aside"],
                             reasons=["the snapshot of %s carries %s: the frozen record is replaced (the old one "
                                      "moved aside)" % (exp, AUDIT_ARTIFACT % exp)])
                continue
            rec["snapshots"] = int(rec.get("snapshots") or 0) + 1
            if rec["snapshots"] >= BUILD_LOST_SNAPSHOTS:
                rec.update(state="gave_up", gave_up_utc=self.utc)
                self._ledger("refreeze_ended", exp=exp, snapshots=rec["snapshots"],
                             reasons=["%d snapshots of %s carried no %s" % (rec["snapshots"], exp,
                                                                            AUDIT_ARTIFACT % exp)])
                self._card("platform", "The label audit of %s does not reach the evidence" % exp,
                           "L4 (proposal %s) ended at %s, but %d snapshots of %s carried no %s (the audit job wrote "
                           "elsewhere, or its file is over INCAP_MAX_FILE_BYTES). D31 does not call for L4 on it again "
                           "(context /audits); a person checks the audit's output." % (
                               rec.get("proposal_id"), rec.get("audit_utc"), rec["snapshots"], exp,
                               AUDIT_ARTIFACT % exp), lever="L4", trigger=["D31"])

    def _observe(self):
        st = self.st
        payload_old = _read_json(self.paths.latest(self.name)) or {}
        status_old = {x.get("exp"): x for x in ((payload_old.get("status") or {}).get("experiments") or [])
                      if isinstance(x, dict)}
        live = self._live_exps()
        ref = (self.dom.get("stage_a") or {}).get("from_exp")
        advance = [e for e in live if e != ref and e in (st.get("exps") or [])
                   and (status_old.get(e) or {}).get("built") and not (status_old.get(e) or {}).get("done")]
        jobs = []
        for ln in ALL_LANES:
            it = st["lanes"][ln].get("item") or {}
            if it.get("status") == "running":
                jobs += [str(j) for j in it.get("job_ids") or []]
        res = X.stream_snapshot(self.sid, exps=live, advance=advance, sacct=jobs,
                                largest=bool(st.get("want_largest")), campaign=self.camp, ctx=self.xctx)
        if res.get("status") == "refused":
            self._once("snapshot_refused", res.get("reasons"), "snapshot_refused", reasons=res.get("reasons"))
            return
        payload = (res.get("remote") or {}).get("payload")
        if not isinstance(payload, dict) or payload.get("verb") != "stream-snapshot":
            st["snapshot_failures"] = int(st.get("snapshot_failures") or 0) + 1
            if st["snapshot_failures"] in (1, 6):
                self._ledger("snapshot_failed", reasons=res.get("reasons"), failures=st["snapshot_failures"])
            if st["snapshot_failures"] >= 6:
                self._card("cluster", "No stream snapshot for %d ticks" % st["snapshot_failures"],
                           "; ".join(res.get("reasons") or []))
            return
        st["snapshot_failures"] = 0
        st["last_snapshot_utc"] = self.utc
        st.pop("await_snapshot", None)
        _write_json(self.paths.latest(self.name), payload)
        self._fold(payload)

    def _fold(self, payload):
        """The snapshot into the lanes: code drift (S23), experiments built and
        done, jobs ended (sacct), sources' intake and admission, holds, health."""
        st = self.st
        # S23: the cluster's stream modules against the lab's
        from . import stream_remote as SR
        lab = SR.module_hashes(LS.CODE_ROOT / "weed_optimizer_framework")
        cl = ((payload.get("code") or {}).get("modules")) or {}
        diff = sorted(k for k in set(lab) | set(cl) if lab.get(k) != cl.get(k))
        if diff:
            if st.get("drift") != diff:
                self._card("drift", "Lab and cluster stream code differ: every stream submission is refused",
                           "modules: %s" % ", ".join(diff[:12]))
                self._ledger("code_drift", modules=diff)
            st["drift"] = diff
        elif st.get("drift"):
            self._ledger("code_drift_cleared", modules=st["drift"])
            st["drift"] = None
        q = ((((payload.get("stream") or {}).get("decision") or {}).get("artifacts")) or {}).get(
            "stream/%s/queue_summary.json" % self.sid) or {}
        self._adopt_fork(q)
        chain = ((payload.get("stream") or {}).get("ledger_chain")) or {}
        if chain.get("ok") is False:
            return self._pause("the stream ledger's hash chain does not verify (%s)" % chain.get("why"))
        status = {x.get("exp"): x for x in ((payload.get("status") or {}).get("experiments") or [])
                  if isinstance(x, dict)}
        sq = (payload.get("status") or {}).get("squeue") or {}
        queued = {str(j.get("id", "")).split("_")[0] for j in sq.get("jobs") or [] if isinstance(j, dict)} \
            if sq.get("ok") else None
        names = {j.get("name") for j in sq.get("jobs") or [] if isinstance(j, dict)} if sq.get("ok") else set()
        sacct = ((payload.get("sacct") or {}).get("jobs")) or {}
        arts = (((payload.get("stream") or {}).get("decision") or {}).get("artifacts")) or {}
        self._arts = arts
        # the experiments the stream built (segments, milestones, Stage C, every
        # bisect arm): tracked, so each is advanced and its status observed
        for e in arts.get("stream/%s/ledger.jsonl" % self.sid) or []:
            if not isinstance(e, dict):
                continue
            built = []                            # (never `names`: the queued job names, read by the items below)
            if e.get("event") in ("build", "milestone", "feasibility") and e.get("phase") in (None, "build"):
                built.append(e.get("exp"))
            if e.get("event") == "bisect" and e.get("phase") == "build":
                built += list((e.get("arms") or {}).values())
            for n_ in built:
                if n_ and NAME_RE.match(str(n_)) and n_ not in st["exps"]:
                    st["exps"].append(str(n_))
        # sources: intake batches (collect.intake summary.json) and the
        # collector's fold of intake/sources.jsonl
        for name, s in sorted(arts.items()):
            if name.startswith("intake/") and name.endswith("/summary.json") and isinstance(s, dict) and s.get("source"):
                src = st["sources"].setdefault(str(s["source"]), {})
                batch = str(s.get("batch") or name.split("/")[1])
                if batch not in (src.get("batches") or []):
                    src.setdefault("batches", []).append(batch)
                    src["batch"] = batch
                src["images"] = s.get("rows")
                src["target_boxes_intake"] = (s.get("yield") or {}).get("target_boxes")
                if s.get("zero_yield"):
                    src["zero_yield_reasons"] = s.get("zero_yield_reasons")
                # the images each batch left deferred (collect.intake step 1b; a batch committed before shards
                # existed records them as yield.images_deferred) and the batch of its fetch before it, so a shard
                # that made no progress is seen (_next_shard)
                sh = s.get("shard") if isinstance(s.get("shard"), dict) else {}
                left = sh.get("deferred_remaining")
                left = int(_num(left if left is not None else (s.get("yield") or {}).get("images_deferred")) or 0)
                earlier = [str(b) for b in sh.get("earlier_batches") or []]
                src.setdefault("shard_log", {})[batch] = {"n": sh.get("n") or 1, "deferred": left,
                                                          "prev": earlier[-1] if earlier else None}
                if batch == src.get("batch"):
                    # 0 once its shards are done (the sweep read the collector's ledger: nothing left to take)
                    src["intake_deferred"] = 0 if src.get("shards_done") == batch else left
                    src["intake_shard"] = sh.get("n")
        srows = arts.get("intake/sources.json") or {}
        if isinstance(arts.get("intake/sources.json"), dict):
            # the cluster collector's fetched facts as of this snapshot, which the
            # byte limits count (executor.fetched_bytes): None when the fold
            # carries none; otherwise a source absent from the ledger fetched
            # nothing on the cluster
            st["cluster_fetched"] = dict(_fetch_facts(srows) or {"sources": None, "open": None}, utc=self.utc)
        _cp, cc = self._collect_config()
        overrides = (cc or {}).get("licence_overrides")
        overrides = overrides if isinstance(overrides, dict) else {}
        for src, row in srows.items():
            if not isinstance(row, dict):
                continue
            s = st["sources"].setdefault(src, {})
            for k in ("bytes", "su", "total_bytes", "failed_attempts"):
                if row.get(k) is not None:
                    s[k] = row[k]
            self._fetch_completeness(src, row.get("fetch_complete"), row.get("fetch_remaining"), "collector ledger")
            rst = row.get("status")
            ov = None
            if rst == "held" and row.get("reason") == "names_pending" and row.get("pending_ts") \
                    and s.get("status") not in ("closed", "quarantined"):
                # an intake refused for class names: the names round, not a hold for a person; a hold
                # already taken up (the same pending_ts) leaves the source where the round has it
                if s.get("names_pending_ts") != row["pending_ts"]:
                    self._names_pending(src, s, row)
                continue
            if rst == "held" and isinstance(overrides.get(src), dict) and (
                    "licence_unresolved" in (row.get("holds") or []) or row.get("reason") == "licence_unresolved"):
                # a person's licence override (the collect config's licence_overrides) lifts the collector's
                # licence hold: collect.plan records that release in the lab's ledger, which no snapshot folds,
                # so a hold in this ledger is read as released here (the next fetch checks the source anew)
                rst, ov = "candidate", overrides[src]
            if rst in ("held", "closed", "quarantined") and s.get("status") not in ("closed", "quarantined"):
                s.pop("names_held", None)          # held now for the collector's own reason, not the names round
                if s.get("status") != rst:
                    self._ledger("source_status", source=src, status=rst, by="collector",
                                 reasons=[_short(row.get("reason") or "", 300)])
                s["status"] = rst
                s["held_by"] = "collector"
                if row.get("reason"):
                    s["held_reason"] = _short(row.get("reason"), 300)
            elif s.get("status") == "held" and s.get("held_by") == "collector" \
                    and rst in ("candidate", "fetched", "intaken"):
                # the collector released its hold (licence, credentials, the copy scan): collectable again; a
                # source part-way through its intake shards waits for its next shard
                s.update(status="shard_pending" if DS.mid_shards(s) else rst, held_by=None, held_reason=None,
                         names_held=None)
                if ov is not None:
                    # the release is the person's decision the collect config records, not the collector's
                    self._ledger("source_released", source=src, status=s["status"], by="licence_overrides",
                                 decided_by=ov.get("decided_by"), decided_utc=ov.get("decided_utc"),
                                 licence=ov.get("id"), research_only=ov.get("research_only"),
                                 collect_config_sha256=LS.sha256_file(_cp) if _cp and Path(_cp).is_file() else None)
                else:
                    self._ledger("source_released", source=src, status=s["status"], by="collector")
            elif rst in ("fetched", "intaken") and s.get("status") in (None, "candidate"):
                # fetched or intaken outside this campaign's own items (a person's run of the collector, a lab
                # fetch, a restarted ticker): the collector's ledger is the record, so DPIPE takes the next step
                # (L16I, then L17 admit) instead of the source waiting with no status for ever; a source part-way
                # through its intake shards takes its next shard
                s["status"] = "shard_pending" if DS.mid_shards(s) else row["status"]
                self._ledger("source_status", source=src, status=s["status"], by="collector",
                             reasons=["adopted from the collector's ledger"])
        self._shard_sweep()
        s1 = (arts.get("step1_stream/status.json") or {}).get("per_source") or {}
        for src, row in s1.items():
            s = st["sources"].setdefault(src, {})
            s["admitted_target_boxes"] = (row or {}).get("target_boxes_admitted", s.get("admitted_target_boxes"))
            if int(_num(s.get("admitted_target_boxes")) or 0) > 0 \
                    and any(r.get("source") == src for r in st.get("zero_run") or []):
                # a source with admitted target boxes did not end with zero yield: an entry of it in the zero-yield
                # run is stale (a failed fetch before its intake, or the first shard of a source still taking
                # shards), and leaves the run now, not at its last shard
                st["zero_run"] = [r for r in st["zero_run"] if r.get("source") != src]
                self._ledger("zero_run_cleared", source=src, admitted_target_boxes=s["admitted_target_boxes"])
            if s.get("status") == "admitted" and not s.get("yield_recorded") and s.get("admit_done_utc"):
                s["yield_recorded"] = True
                boxes = int(s.get("admitted_target_boxes") or 0)
                self._zero_yield(src, "yield" if boxes else "zero", boxes)
        # items
        for ln in ALL_LANES:
            it = st["lanes"][ln].get("item")
            if not it or it.get("status") != "running" or it.get("lab_job"):
                continue
            p = it["proposal"]
            follow = p.get("follow")
            child = p.get("child_exp")
            if follow in ("build", "experiment") and child:
                self._settle_build_job(it, queued, sacct)
                x = status.get(child) or {}
                if x.get("built"):
                    for s in st["segments"]:
                        if s["exp"] == child and not s.get("built_utc"):
                            s["built_utc"] = self.utc
                    if follow == "build" or (follow == "experiment" and x.get("done") and x.get("report") == "current"):
                        self._on_exp_done(child, x) if x.get("done") else None
                        self._done(ln, it)
                        continue
                    if follow == "experiment":
                        continue
                ended = queued is not None and not any(j.split("_")[0] in queued for j in it.get("job_ids") or []) \
                    and ("inc_build_%s" % child) not in names
                if ended:
                    it["lost"] = int(it.get("lost") or 0) + 1
                    states = [(sacct.get(j) or {}).get("state") for j in it.get("job_ids") or []]
                    if it["lost"] >= BUILD_LOST_SNAPSHOTS or any(s_ not in (None, "RUNNING", "PENDING", "COMPLETED")
                                                                for s_ in states):
                        prov = self._build_prov(payload, it, child)
                        # the build ran and ended without its experiment: what _failed needs to tell a refusal
                        # of the build itself from a job killed from outside (a measurement arm's build)
                        why = "the build of %s ended without the experiment%s" % (
                            child, (": %s" % prov.get("refusal")) if prov.get("refusal") else "")
                        ended = {"refusal": prov.get("refusal"), "states": [str(s_) for s_ in states if s_]}
                        if it["lever"] == "L23I":
                            # a family build that ended without its experiment: its build job's GPU-h count against
                            # the family cap (decision 25), never 0
                            self._v3fam_settle(it, [sacct.get(j) for j in it.get("job_ids") or []], build_ended=True)
                        if it["lever"] == "L32":
                            self._ds1_settle(it, [sacct.get(j) for j in it.get("job_ids") or []], build_ended=True)
                        if not (it["lever"] == "L18" and self._cut_short(ln, it, why, ended, q)):
                            self._failed(ln, it, why, ended=ended)
                continue
            if follow == "job":
                ids = [str(j) for j in it.get("job_ids") or []]
                if not ids and it.get("uncertain") and it["lever"] in RECORD_ONLY_LEVERS:
                    self._follow_record_only(ln, it, queued, names, arts, payload=payload)
                    continue
                if queued is None or any(j.split("_")[0] in queued for j in ids):
                    if it["lever"] == "L23K":
                        self._v3fam_overdue(it)
                    if it["lever"] == "L32X":
                        self._ds1_overdue(it)
                    continue
                states = [((sacct.get(j) or {}).get("state") or "") for j in ids]
                if ids and not all(states):
                    it["lost"] = int(it.get("lost") or 0) + 1
                    if it["lost"] < BUILD_LOST_SNAPSHOTS:
                        continue
                ok = bool(ids) and all(s.startswith("COMPLETED") for s in states)
                # the refusal line of the job's log (stream_remote.sacct: data jobs only)
                refusal = "" if ok else "; ".join(str((sacct.get(j) or {}).get("refusal")) for j in ids
                                                  if (sacct.get(j) or {}).get("refusal"))
                if not ok and not refusal and it["lever"] in V3FAM_LEVERS:
                    # a family job: its provenance's last attempt when it is this job's (deviation 5: how a
                    # precondition of the platform's own making is told from the arm's own failure)
                    refusal = self._v3fam_prov_refusal(payload, it, ids)
                if p["policy_action"] not in ("inc_label_audit",):
                    for j in ids:
                        if sacct.get(j):
                            B.record_job_spend(j, self.name, sacct[j], p["policy_action"], domain=self.domain,
                                               base_dir=self.xctx.su_base_dir, ts=self.utc)
                if it["lever"] in V3FAM_LEVERS:
                    self._v3fam_settle(it, [sacct.get(j) for j in ids])
                    it["v3fam_states"] = states
                if it["lever"] in DS1_LEVERS:
                    self._ds1_settle(it, [sacct.get(j) for j in ids])
                    it["ds1_states"] = states
                s_ = st["sources"].get((p.get("params") or {}).get("source")) or {}
                if not ok and it["lever"] == "L16I" and (s_.get("status") == "names_pending" or (
                        s_.get("status") == "held" and s_.get("names_held")) or NAMES_REFUSAL in refusal):
                    self._intake_names_pending(ln, it, ids, states)
                    continue
                if ok:
                    self._done(ln, it)
                    continue
                why = "job(s) %s ended %s%s" % (",".join(ids), ",".join(states) or "unknown",
                                                ("; refused: %s" % refusal) if refusal else "")
                m = COLLECT_REFUSAL_RE.search(refusal or "")
                ref = self._collector_refusal(it, 2, {"code": m.group(1), "detail": m.group(2)}) if m and all(
                    str(x).startswith("FAILED") for x in states) else None
                if ref is not None:
                    self._source_refused(ln, it, ref, why)
                else:
                    self._failed(ln, it, why, charged=True, retry=not (p["policy_action"] == "inc_stream_admit"
                                                                      and RETRY_FALSE_RE.search(refusal)))
        # a fork (L22) whose job this very snapshot shows ended: its summary
        # already names the new stream, so it is adopted now, not one tick
        # later on evidence that still reads the old stream's last commit
        self._adopt_fork(q)
        # experiments the stream reads: done -> frozen (spend recorded for its own)
        for exp in list(dict.fromkeys(list(st.get("exps") or []) + self._r0_exps())):
            x = status.get(exp) or {}
            if x.get("done") and x.get("report") == "current":
                self._on_exp_done(exp, x)
                sub = (payload.get("experiments") or {}).get(exp) or {}
                snap = sub.get("snapshot")
                if isinstance(snap, dict) and snap.get("built") and exp not in (st.get("frozen") or {}):
                    _write_json(self.paths.frozen(self.name, exp),
                                {"exp": exp, "frozen_utc": self.utc,
                                 "snapshot": {k: v for k, v in snap.items() if k != "display_only"}})
                    st.setdefault("frozen", {})[exp] = {"utc": self.utc}
        # a frozen experiment snapshotted again for the label audit run on it after its freeze
        self._settle_refreeze(payload)
        self._health(payload)

    def _build_prov(self, payload, it, child):
        """The last attempt of the build job's provenance record (the
        snapshot's builds, remote.status): the experiment's own record
        (run_inc2_build.sh names a baseline's after its --exp), else the
        stream's when its last attempt is this item's job: every inc2.stream
        build of a stream writes provenance/stream_<sid>.json (a segment's
        refusal was looked up under the segment's name and never found)."""
        builds = (payload.get("status") or {}).get("builds") or {}
        if isinstance(builds.get(child), dict) and builds[child]:
            return builds[child]
        sid = ((it["proposal"].get("params") or {}).get("stream")) or self.sid
        rec = builds.get("stream_%s" % sid)
        jobs = {str(j).split("_")[0] for j in it.get("job_ids") or []}
        return rec if isinstance(rec, dict) and str(rec.get("job_id") or "") in jobs else {}

    def _cut_short(self, ln, it, why, ended, q):
        """True when a segment build (L18) the cutter refused for want of
        eligible images (SHORT_RE) was handled here. While the queue summary
        this snapshot shipped holds fewer than M eligible images the refusal
        is no failure: the queue fell after D22 called for the cut (a
        quarantine, rows consumed or held), so it is recorded not_taken, its
        id retired (the next cut runs under a new one), its build's estimate
        released, and neither the lane's failure count nor the step's is
        touched. While Q >= M (or Q unknown) the cutter and D22 disagree: a
        failure as before, and a card. Live, 2026-10-05: L24 quarantined
        zenodo_15808623 in the tick L18 was proposed, Q fell from 8,041 to
        319 (M 1,364), and the cutter's 'short' counted as TRAIN's failed step."""
        if not SHORT_RE.search(str(ended.get("refusal") or "")):
            return False
        st, p = self.st, it["proposal"]
        Q = _num(((q or {}).get("eligible") or {}).get("images"))
        Mv = _num((q or {}).get("M")) or _num((self.dom.get("increment") or {}).get("M"))
        if Q is not None and Mv and Q < Mv:
            self._build_ended_spend(it, why)
            st["failed_ids"] = (self._failed_ids() + [p["id"]])[-FAILED_IDS_KEEP:]
            self._ledger("not_taken", lane=ln, lever=it["lever"], proposal_id=p["id"], child_exp=p.get("child_exp"),
                         Q=int(Q), M=int(Mv), job_ids=list(it.get("job_ids") or []), states=ended.get("states"),
                         reasons=["%s; the queue now holds Q %d < M %d eligible images, so the cut was refused for "
                                  "want of data, not failed: no lane failure, a new id for the next cut"
                                  % (_short(why, 300), Q, Mv)])
            self._clear(ln)
            return True
        self._card("escalation", "Segment %s: the cutter refused it short while the queue holds Q %s >= M %s"
                   % (p.get("child_exp"), "unknown" if Q is None else "%d" % Q, "%d" % Mv if Mv else "unknown"),
                   "%s. D22 counts Q from the queue summary and the cutter found fewer eligible images than M: the "
                   "two disagree (a guard exclusion, a stale summary). Counted as a failed step of TRAIN; a person "
                   "reads the cutter's last_refusal in the queue summary." % _short(why, 400),
                   lever=it["lever"], trigger=list(p.get("trigger") or []))
        self._failed(ln, it, why, ended=ended)
        return True

    def _record_only_follow(self, it):
        """(job name, record artifact) a record-only item is followed by (an instance method: L23L's record is the
        domain's verdict record)."""
        from . import stream_remote as SR
        prm = it["proposal"].get("params") or {}
        if it["lever"] == "L23V":
            return SR.BASE3_JOB_NAME, "splits/v3/summary.json"
        if it["lever"] == "L23E":
            return SR.AGNOSTIC_JOB_NAME % prm.get("exp"), "%s/agnostic_rescore.json" % prm.get("exp")
        if it["lever"] == "L23C":
            return SR.E2_JOB_NAME, "capacity/e2_rescore.json"
        if it["lever"] == "L23D":
            return SR.E2_ATTR_JOB_NAME, "capacity/e2_attr_rescore.json"
        if it["lever"] == "L23Z":
            return tuple(n % prm.get("version", "v1") for n in SR.ZOO_JOB_NAMES), "capacity/zoo_%s.json" % prm.get(
                "version", "v1")
        if it["lever"] == "L23F":
            arm = str(prm.get("arm"))
            return SR.E3_SCORE_JOB_NAME % arm.lower(), "capacity/e3_score_%s.json" % arm
        if it["lever"] == "L23G":
            return SR.E3_JOB_NAME, "capacity/e3_rescore.json"
        if it["lever"] == "L23H":
            return SR.V3FAM_DATA_JOB_NAME, "splits/v3fam/summary.json"
        if it["lever"] == "L23J":
            arm = str(prm.get("arm"))
            return SR.V3FAM_SCORE_JOB_NAME % arm.lower(), "capacity/v3fam_score_%s.json" % arm.lower()
        if it["lever"] == "L23K":
            return tuple(SR.V3FAM_DETR_JOB_NAMES[str(prm.get("verb"))]), SR.V3FAM_STATUS
        if it["lever"] in DS1_LEVERS:
            return self._ds1_follow(it)
        if it["lever"] == "L23L":
            # the domain's verdict record: capacity/v3fam_v1_cont.json once the verdict of 2026-10-13 was in place
            # (deviation 4), capacity/v3fam_v1_cont2.json once the continuation of 2026-10-22 is (deviation 5); the
            # job's name is the same
            return SR.V3FAM_JOB_NAME, (((self.dom or {}).get("v3fam") or {}).get("verdict_record")
                                       or "capacity/v3fam_v1.json")
        return SR.NATIVE_JOB_NAME % prm.get("exp"), "%s/native_rescore.json" % prm.get("exp")

    def _follow_record_only(self, ln, it, queued, names, arts, payload=None):
        """A record-only item (L23N, L23V, L23E, L23C, L23D, L23Z, L23F,
        L23G) whose submission's outcome is unknown: it runs while a job of
        its name (L23Z: any of its chain's five) is queued; then it is done
        once its record (<exp>/native_rescore.json, splits/v3/summary.json,
        <exp>/agnostic_rescore.json, capacity/e2_rescore.json,
        capacity/e2_attr_rescore.json, capacity/zoo_v1.json,
        capacity/e3_score_<arm>.json, capacity/e3_rescore.json) says complete
        (the zoo's: complete or partial), and failed (record only: a card)
        after BUILD_LOST_SNAPSHOTS snapshots with neither."""
        job, record = self._record_only_follow(it)
        jobs = job if isinstance(job, tuple) else (job,)
        if queued is None or any(j in names for j in jobs):
            return
        rs = (arts.get(record) or {}).get("status")
        if it["lever"] in V3FAM_LEVERS:
            # V3F: its records' own words for done (the verdict's 'decided'; L23K's from the derived status)
            rs = "complete" if self._v3fam_record_done(it, arts.get(record) or {}) else None
        if it["lever"] in DS1_LEVERS:
            rs = "complete" if self._ds1_record_done(it, arts.get(record) or {}) else None
        if rs == "complete" or (it["lever"] == "L23Z" and rs == "partial"):
            self._done(ln, it)
            return
        if it["lever"] == "L23Z" and rs in ("refused", "failed"):
            self._failed(ln, it, "the zoo record says %s%s" % (rs, (": %s" % _short((arts.get(record) or {}).get(
                "refusal"), 300)) if (arts.get(record) or {}).get("refusal") else ""), charged=True)
            return
        it["lost"] = int(it.get("lost") or 0) + 1
        if it["lost"] >= BUILD_LOST_SNAPSHOTS:
            why = "its submission's outcome was unknown, and in %d snapshots no job named %s was queued and %s was " \
                  "not complete" % (it["lost"], job, record)
            if it["lever"] in V3FAM_LEVERS:
                # a family job that ran after the submission: its provenance's refusal (deviation 5), so a
                # precondition of the platform's own making is never counted as the arm's failure here either
                ref = self._v3fam_prov_refusal(payload, it, None)
                if ref:
                    why += "; refused: %s" % ref
            self._failed(ln, it, why, charged=True)

    def _settle_build_job(self, it, queued, sacct):
        """6.6: the build job itself (run_inc2_build.sh holds one V100 on
        GPU-shared while it builds) is settled from sacct once it has ended;
        the experiment's runs are settled by its report spend."""
        if it.get("build_settled") or queued is None:
            return
        ids = [str(j) for j in it.get("job_ids") or []]
        if not ids or any(j.split("_")[0] in queued for j in ids):
            return
        rows = {j: sacct.get(j) for j in ids}
        if not all(r and r.get("state") and not str(r["state"]).startswith(("RUNNING", "PENDING"))
                   for r in rows.values()):
            return
        for j, r in rows.items():
            B.record_job_spend(j, self.name, r, it["proposal"].get("policy_action"), domain=self.domain,
                               base_dir=self.xctx.su_base_dir, ts=self.utc)
        it["build_settled"] = self.utc

    def _build_ended_spend(self, it, why):
        """A build whose job ran and ended without its experiment: once
        _settle_build_job wrote its jobs' sacct spend, each job is marked
        ended (budget.record_build_ended), so budget.committed releases the
        build's estimate, which only its experiment's report spend releases
        otherwise and which would then stay committed for good (the job's
        sacct SU in spent as well). A build sacct did not settle keeps its
        estimate: overstated, never understated."""
        if not it.get("build_settled"):
            return
        p = it["proposal"]
        child = p.get("child_exp") or (p.get("params") or {}).get("exp")
        for j in [str(x) for x in it.get("job_ids") or []]:
            r = B.record_build_ended(j, self.name, child, _short(why, 300), domain=self.domain,
                                     base_dir=self.xctx.su_base_dir, ts=self.utc)
            if not r.get("ok"):
                self.log.warning("[inc-stream] build job %s of %s: its estimate stays committed (%s)"
                                 % (j, child, r.get("reason")))

    def _adopt_fork(self, q):
        """The new stream version a finished fork (L22) created: the campaign's
        sid moves to it (its summary names it as forked_to)."""
        st = self.st
        if not (st.get("fork_pending") and NAME_RE.match(str((q or {}).get("forked_to") or ""))):
            return
        new_sid = str(q["forked_to"])
        old_sid = self.sid
        from . import campaign as C

        def change(c):
            c.setdefault("stream", {})["sid"] = new_sid
            return c
        try:
            C._update_config(self.cfg_hooks, self.name, change)
            self.cfg.setdefault("stream", {})["sid"] = new_sid
            # items proposed or filed for the old stream (it builds nothing
            # more) are withdrawn, not left to fail against it
            for ln in ALL_LANES:
                it = st["lanes"][ln].get("item") or {}
                prm = (it.get("proposal") or {}).get("params") or {}
                if it and it.get("status") in ("proposed", "filed") and (
                        prm.get("stream") == old_sid or str(prm.get("exp") or "").startswith(old_sid + "_")):
                    self._ledger("withdrawn", lane=ln, lever=it.get("lever"), approval_id=it.get("approval_id"),
                                 reasons=["the stream %s was forked to %s" % (old_sid, new_sid)])
                    st["declined"] = (list(st.get("declined") or []) + [it["proposal"]["id"]])[-200:]
                    self._clear(ln)
            self._ledger("fork_adopted", sid=new_sid, previous_sid=q.get("sid"), trigger=["D8S"],
                         reasons=["L22 doubled M: a new stream version adopts the pool and the quarantine"])
            st["fork_pending"] = None
            st["segments"], st["milestones"] = [], []
            # the evidence in hand is the old stream's: nothing is proposed on
            # it for the new one (its init would read as missing, and LI would
            # be proposed); the next tick's one call is the new stream's snapshot
            st["await_snapshot"] = new_sid
        except Exception as e:
            self.log.warning("[inc-stream] %s: the fork's sid not written: %s" % (self.name, e))

    def _on_exp_done(self, exp, x):
        st = self.st
        for s in st["segments"]:
            if s["exp"] == exp and not s.get("done"):
                s["done"] = True
                s["done_utc"] = self.utc
                self._ledger("segment_done", exp=exp)
        for m in st["milestones"]:
            if m["exp"] == exp and not m.get("done"):
                m["done"] = True
                q = self._artifact("stream/%s/queue_summary.json" % self.sid) or {}
                m["pool"] = (q.get("pool") or {}).get("current") or m.get("pool")
                self._ledger("milestone_done", exp=exp, pool=m["pool"])
        reported = st.setdefault("reported", {})
        if exp in reported or exp not in (st.get("exps") or []):
            return
        rep = self.ev.json("%s/report.json" % exp) if self.ev is not None else None
        if isinstance(rep, dict) and rep.get("done"):
            r = B.record_report_spend(rep, self.name, domain=self.domain, base_dir=self.xctx.su_base_dir)
            reported[exp] = {"utc": self.utc, "su": r.get("su"), "ok": r.get("ok")}
            self._ledger("reported", exp=exp, spend={k: r.get(k) for k in ("ok", "su", "reason")})

    def _health(self, payload):
        """D5-D7 and D14 on every live experiment of the snapshot (the pinned
        driver's experiments: the autopilot's own health rules, diagnose.py),
        and the stops they call for. A measurement arm's (a baseline marked
        measure: record only, no lane waits for it) D5 and D6 never stop the
        stream: they are a person's card (_record_only)."""
        st = self.st
        out = []
        measure = self._measure_exps()
        sq = (payload.get("status") or {}).get("squeue") or {}
        names = [j.get("name") for j in sq.get("jobs") or [] if isinstance(j, dict)] if sq.get("ok") else None
        for exp, sub in sorted((payload.get("experiments") or {}).items()):
            snap = (sub or {}).get("snapshot") or {}
            if not snap.get("built") or exp in (st.get("refreeze") or {}):
                continue                         # a frozen experiment snapshotted again for its audit only
            hist = st.setdefault("history", {}).setdefault(exp, [])
            gen = (((snap.get("decision") or {}).get("artifacts") or {}).get("%s/state.json" % exp) or {}).get("generation")
            if gen is not None:
                hist.append({"utc": self.utc, "generation": gen})
                st["history"][exp] = hist[-48:]
            ctx = {"now_utc": self.utc, "history": list(st["history"][exp])}
            if names is not None:
                ctx["squeue"] = names
            adv = (sub or {}).get("advance")
            if isinstance(adv, dict) and adv.get("ok") is False and adv.get("error"):
                ctx["advance"] = {"error": _short(adv.get("error"), 2000), "error_kind": adv.get("error_kind")}
            rec = {"verb": "campaign-snapshot", "experiments": {exp: {"snapshot": {k: v for k, v in snap.items()
                                                                                    if k != "display_only"}}}}
            try:
                ev = E.from_snapshot(rec, exp, context=ctx, domain=LS.funnel_ref(self.dom))
            except Exception as e:
                self._once("health_ev:%s" % exp, str(e), "evidence_error", exp=exp, error=_short(e, 300))
                continue
            for d in DG.detect(ev, only=HEALTH_EXP):
                if d.get("fired"):
                    out.append(_record_only(d) if exp in measure and d["id"] == "D5" else d)
        st["health"] = out
        by = {}
        for d in out:
            by.setdefault(d["id"], []).append(d)
        for did, what in (("D14", "halt: a decision path touched a non-dev exam"),
                          ("D7", "code drift: the driver refused to advance on code")):
            if did in by:
                if did == "D7":
                    self._card("research", "X5: repin or sync the outer copy", by[did][0].get("summary"), lever="X5",
                               trigger=[did])
                return self._pause("%s (%s): %s" % (what, did, _short(by[did][0].get("summary"), 400)))
        for d in by.get("D5") or []:
            if "OP_PAUSE" in (d.get("levers") or []):
                return self._pause("a blocked unit of %s is not transient (D5): %s" % (d.get("exp"),
                                                                                   _short(d.get("summary"), 300)))
            if (d.get("detail") or {}).get("record_only"):
                self._measure_card(d)
            stop = st["lanes"]["STOP"]
            units = [u for u in (d.get("detail") or {}).get("units") or [] if u.get("transient")]
            if units and "L7" in (d.get("levers") or []) and stop.get("item") is None:
                u = units[0]
                argv = ["python", "-m", "weed_optimizer_framework.tools.inc.driver", "unblock", "--exp", d.get("exp"),
                        "--unit", u["unit"], "--reason", "auto: %s" % u["cause"]]
                p = {"id": _sha([self.name, "L7", argv])[:32], "lever": "L7", "family": "L7", "argv": argv,
                     "params": {"exp": d.get("exp"), "unit": u["unit"], "cause": u["cause"]},
                     "policy_action": "inc_unblock_transient", "risk": "R2", "trigger": ["D5"],
                     "cites": list(d.get("cites") or []), "lit": [], "control": "", "success": "", "falsifier": "",
                     "est_gpu_hours": 0.0, "proposed_by": AUTO, "lane": "STOP", "follow": "login",
                     "parent_exp": d.get("exp"), "child_exp": None, "attempt": 0}
                if p["id"] not in (st.get("declined") or []):
                    stop["item"] = {"proposal": p, "lever": "L7", "status": "proposed", "since_utc": self.utc,
                                    "attempt": 0, "trigger": "D5", "diagnoses": [d]}
                    self._ledger("proposed", lane="STOP", lever="L7", action="inc_unblock_transient", risk="R2",
                                 argv=argv, trigger=["D5"], proposal_id=p["id"])
        stale = st.setdefault("stale", {})
        for d in by.get("D6") or []:
            stale[d["exp"]] = int(stale.get(d["exp"]) or 0) + 1
            if stale[d["exp"]] >= STALE_TO_PAUSE:
                if d["exp"] in measure:
                    self._measure_card(d)
                    continue
                return self._pause("stale advance (D6) of %s on %d consecutive snapshots" % (d["exp"], stale[d["exp"]]))
        for exp in list(stale):
            if not any(d.get("exp") == exp for d in by.get("D6") or []):
                stale[exp] = 0
        # DS1 (2026-10-29): a DS1 build whose runs are blocked (D5, not transient) or stale (D6) is recorded so its
        # arm counts as failed in DS1's plan (DS.9.3: a failed arm never holds the other arms); cleared once the
        # experiment advances again (a person unblocked it)
        self._ds1_runs_blocked([d for d in by.get("D5") or [] if (d.get("detail") or {}).get("record_only")] +
                               [d for d in by.get("D6") or [] if int(stale.get(d["exp"]) or 0) >= STALE_TO_PAUSE])

    def _zoo_stage(self, payload, r0):
        """/stage/zoo: done once capacity/zoo_<v>.json says complete; partial;
        failed (refused, failed, or the platform's item failed); stale (a live
        status with no zoo job queued and no item of the platform following
        it: a card, never a second proposal); running; else what the platform
        ran (unconfirmed when its item ended without a record), else missing."""
        from . import stream_remote as SR
        z = (self.dom or {}).get("zoo") or {}
        rec = self._artifact(z.get("record") or "capacity/zoo_v1.json") or {}
        rs = rec.get("status")
        self._zoo_rec = rec
        out = None
        if rs == "complete":
            out = "done"
        elif rs == "partial":
            out = "partial"
        elif rs in ("refused", "failed"):
            out = "failed"
        elif rs:
            if r0.get("zoo") in ("failed", "running"):
                out = r0["zoo"]
            else:
                out = "running"
                sq = ((payload or {}).get("status") or {}).get("squeue") or {}
                if sq.get("ok"):
                    names = {j.get("name") for j in sq.get("jobs") or [] if isinstance(j, dict)}
                    v = rec.get("version") or z.get("version") or "v1"
                    if not any((n % v) in names for n in SR.ZOO_JOB_NAMES):
                        out = "stale"
        elif r0.get("zoo") == "done":
            out = "unconfirmed"
        else:
            out = r0.get("zoo") or "missing"
        self._zoo_state = out
        return out

    def _zoo_rerun(self, params):
        z = (self.dom or {}).get("zoo") or {}
        g = lambda k: (params or {}).get(k) or z.get(k)  # noqa: E731
        return ZOO_RERUN % {"v": g("version") or "v1", "a": g("shards_a"), "c": g("shards_c"),
                            "p": g("concurrency"), "h": g("max_gpu_hours")}

    def _zoo_card(self, params, p):
        """The zoo's card once its item is done: counts and the report's path
        (a research card when complete; an escalation naming the failed shards
        and the rerun commands when partial). Never a metric."""
        z = (self.dom or {}).get("zoo") or {}
        v = (params or {}).get("version") or z.get("version") or "v1"
        rec = self._artifact(z.get("record") or "capacity/zoo_%s.json" % v) or {}
        c = rec.get("counts") or {}
        sk = ", ".join("%s %s" % (k, n) for k, n in sorted((c.get("skipped_by_reason") or {}).items()) if n)
        lines = ["files listed %s: kept %s, unscorable %s, skipped %s (%s); converted %s"
                 % (c.get("files_listed"), c.get("kept"), c.get("unscorable"), c.get("skipped"), sk or "-",
                    c.get("converted")),
                 "items: planned %s, scored %s, reused %s, refused %s, errors %s, not scored (budget) %s, not scored "
                 "(time) %s" % (c.get("items_planned"), c.get("items_scored"), c.get("items_reused"),
                                c.get("items_refused"), c.get("items_error"), c.get("items_not_scored_budget"),
                                c.get("items_not_scored_time")),
                 "report: INC_DIR/_zoo/%s/report.md (report.json, report.csv; the sealed reads in report_external.*)"
                 % v, "descriptive: no model is adopted or called best from it (Amendment Z1)."]
        if rec.get("status") == "complete":
            self._card("research", "Model-zoo audit %s complete" % v, "\n".join(lines), lever="L23Z",
                       trigger=list((p or {}).get("trigger") or []))
            return
        sh = rec.get("shards") or {}
        failed = ["%s%s" % (k, (sh.get(k) or {}).get("failed")) for k in ("a", "c") if (sh.get(k) or {}).get("failed")]
        self._card("escalation", "Model-zoo audit %s partial (L23Z)" % v,
                   "\n".join(lines + ["record status %s; failed shards %s. A person reruns: %s"
                                       % (rec.get("status"), ", ".join(failed) or "none named", self._zoo_rerun(params))]),
                   lever="L23Z", trigger=list((p or {}).get("trigger") or []))

    def _zoo_stale_card(self):
        """One card per stale zoo record (/stage/zoo stale): a chain that
        stopped part-way, followed by nobody."""
        st = self.st
        rec = getattr(self, "_zoo_rec", None) or {}
        if getattr(self, "_zoo_state", None) != "stale" or st.get("zoo_stale_card") == rec.get("updated_utc"):
            return
        st["zoo_stale_card"] = rec.get("updated_utc")
        self._card("escalation", "The model-zoo audit stopped part-way (record %s, no zoo job queued)"
                   % rec.get("status"), "capacity/zoo_%s.json says %s and no inc_zoo job is queued. A person "
                   "resubmits (resumable): %s" % (rec.get("version") or "v1", rec.get("status"), self._zoo_rerun({})),
                   lever="L23Z", trigger=["L23Z"])

    def _measure_build(self, lever, params, p):
        """The measurement baseline an L23B item builds (a baselines item
        marked measure, by its child experiment), or None."""
        if lever != "L23B":
            return None
        child = p.get("child_exp") or (params or {}).get("exp")
        return next((b for b in ((self.dom or {}).get("baselines") or {}).get("items") or []
                     if b.get("measure") and b.get("exp") == child), None)

    def _e2_items(self):
        """{'W': [items], 'S': [items]} of E2's baselines (stream-domain e2.arms), or {}."""
        e2 = (self.dom or {}).get("e2") or {}
        items = {b["id"]: b for b in ((self.dom or {}).get("baselines") or {}).get("items") or []}
        return {k: [items[i] for i in ids if i in items] for k, ids in sorted((e2.get("arms") or {}).items())}

    def _e2_card(self):
        """One card once E2's verdict (capacity/e2_v1.json, shipped) is
        decided: the arms that qualify, each with its D, 2 pooled sd and SE,
        and for each qualifying arm the exact commands of a person's one read
        of its sealed test (the chosen arm's is E2's headline). Raised once
        (state e2_card)."""
        st = self.st
        if st.get("e2_card"):
            return
        rec = ((self.dom or {}).get("e2") or {}).get("record") or "capacity/e2_v1.json"
        v = self._artifact(rec) or {}
        if v.get("status") != "decided":
            return
        q = [k for k in v.get("qualifying") or [] if k in ("W", "S")]
        chosen = v.get("chosen")
        lines = []
        for k in ("W", "S"):
            a = (v.get("arms") or {}).get(k) or {}
            if a.get("status") == "decided":
                lines.append("E2-%s: D %s, 2 x pooled sd %s, SE %s -> %s" % (
                    k, _fmt4(a.get("diff")), _fmt4(a.get("two_pooled_sd")), _fmt4(a.get("se_diff")),
                    "qualifies" if a.get("qualifies") else "does not qualify"))
        if q:
            title = "E2's verdict: E2-%s qualif%s" % (", E2-".join(q), "ies" if len(q) == 1 else "y")
            if chosen in q:
                title += " (chosen: E2-%s)" % chosen
                lines.append("E2's choice is E2-%s (the larger D; a tie goes to S): its sealed test is E2's headline "
                             "number. Nothing switches." % chosen)
            for k in q:
                lines.append("A person reads E2-%s's sealed test once: python -m weed_optimizer_framework.tools.inc2."
                             "baseline e2-test-read --e2 %s, submit the printed argvs, then python -m "
                             "weed_optimizer_framework.tools.inc2.baseline e2-test-report --e2 %s." % (k, k, k))
        else:
            lines.append("No arm qualifies: nothing to read; nothing switches.")
            title = "E2's verdict: no arm qualifies"
        self._card("research", title, "\n".join(lines), trigger=["L23C"])
        st["e2_card"] = self.utc

    def _e2_attr_items(self):
        """{'C': [items], 'W': [items]} of E2-C's attribution (stream-domain e2_attr.arms), or {}."""
        ea = (self.dom or {}).get("e2_attr") or {}
        items = {b["id"]: b for b in ((self.dom or {}).get("baselines") or {}).get("items") or []}
        return {k: [items[i] for i in ids if i in items] for k, ids in sorted((ea.get("arms") or {}).items())}

    def _e2_attr_card(self):
        """One card once E2-C's attribution (capacity/e2_attr_v1.json,
        shipped) is decided: D_data with 2 pooled sd and SE and what it is
        credited to, E2-S - E2-C (confounded by the recipe) when E2 chose S,
        E2-C - the reference beside, and, when E2 has a qualifying choice,
        the exact commands of a person's one read of E2-C's sealed test, with
        the chosen arm's and beside it. Raised once (state e2_attr_card)."""
        st = self.st
        if st.get("e2_attr_card"):
            return
        rec = ((self.dom or {}).get("e2_attr") or {}).get("record") or "capacity/e2_attr_v1.json"
        a = self._artifact(rec) or {}
        if a.get("status") != "decided":
            return
        d = a.get("data") or {}
        credited = a.get("credited_to_data") is True
        lines = ["D_data = E2-W - E2-C %s, 2 x pooled sd %s, SE %s -> %s" % (
            _fmt4(d.get("diff")), _fmt4(d.get("two_pooled_sd")), _fmt4(d.get("se_diff")),
            "E2's gain is credited to base v3's data" if credited
            else "E2's result is a property of one-class pre-training, not of data scale")]
        c = a.get("chosen_vs_control") or {}
        if c:
            lines.append("E2-S - E2-C %s, 2 x pooled sd %s, SE %s (confounded by the recipe: reported only)" % (
                _fmt4(c.get("diff")), _fmt4(c.get("two_pooled_sd")), _fmt4(c.get("se_diff"))))
        r = ((a.get("reported") or {}).get("control_vs_reference")) or {}
        lines.append("Beside: %s %s, SE %s (two-stage pre-training without extra data)" % (
            r.get("comparison") or "E2-C - b_v2_m640", _fmt4(r.get("diff")), _fmt4(r.get("se_diff"))))
        v = a.get("e2_verdict") or {}
        chosen = v.get("chosen")
        if chosen and chosen in (v.get("qualifying") or []):
            lines.append("A person reads E2-C's sealed test once, with E2-%s's and after its read is prepared, beside "
                         "the headline: python -m weed_optimizer_framework.tools.inc2.baseline e2-test-read --e2 C, "
                         "submit the printed argvs, then python -m weed_optimizer_framework.tools.inc2.baseline "
                         "e2-test-report --e2 C. Nothing switches; capacity/e2_v1.json is unchanged." % chosen)
        else:
            lines.append("No E2 arm qualifies: E2-C's test is not read; nothing switches.")
        title = "E2-C's attribution: %s" % ("credited to base v3's data" if credited else
                                            "one-class pre-training, not data scale")
        self._card("research", title, "\n".join(lines), trigger=["L23D"])
        st["e2_attr_card"] = self.utc

    def _e3_card(self):
        """One card once E3's verdict (capacity/e3_v1.json, shipped) is
        decided: each arm's D, 2 pooled sd, SE and whether it qualifies, the
        choice, the attribution (E3-B - E3-A) and what it is credited to,
        the classifier's dev ground-truth top-1 and each arm's stage-1
        agnostic AP, and for each qualifying arm the exact commands of a
        person's one read of its sealed test (reported against 0.8786, the
        gap to 0.90; the chosen arm's is E3's headline). Raised once (state
        e3_card)."""
        st = self.st
        if st.get("e3_card"):
            return
        rec = ((self.dom or {}).get("e3") or {}).get("record") or "capacity/e3_v1.json"
        v = self._artifact(rec) or {}
        if v.get("status") != "decided":
            return
        order = list(((self.dom or {}).get("e3") or {}).get("order") or ["M", "A", "B"])
        q = [k for k in v.get("qualifying") or [] if k in order]
        chosen = v.get("chosen")
        lines = []
        for k in order:
            a = (v.get("arms") or {}).get(k) or {}
            if a.get("status") == "decided":
                s1 = ((a.get("reported") or {}).get("stage1_agnostic") or {}).get("mean")
                lines.append("E3-%s: D %s, 2 x pooled sd %s, SE %s -> %s (stage-1 agnostic %s)" % (
                    k, _fmt4(a.get("diff")), _fmt4(a.get("two_pooled_sd")), _fmt4(a.get("se_diff")),
                    "qualifies" if a.get("qualifies") else "does not qualify", _fmt4(s1)))
        at = v.get("attribution") or {}
        if at.get("status") == "decided":
            lines.append("Attribution E3-B - E3-A: D_data %s, 2 x pooled sd %s, SE %s -> %s" % (
                _fmt4(at.get("diff")), _fmt4(at.get("two_pooled_sd")), _fmt4(at.get("se_diff")),
                "E3's box gain is credited to base v3's data" if at.get("credited_to_data")
                else "not credited to base v3's data"))
        dg = (v.get("reported") or {}).get("dev_gt") or {}
        if dg.get("top1") is not None:
            lines.append("The classifier on dev's ground-truth crops: top-1 %s (%s crops)" % (_fmt4(dg.get("top1")),
                                                                                         dg.get("n")))
        if q:
            title = "E3's verdict: E3-%s qualif%s" % (", E3-".join(q), "ies" if len(q) == 1 else "y")
            if chosen in q:
                title += " (chosen: E3-%s)" % chosen
                lines.append("E3's choice is E3-%s (the largest D; a tie goes to M, then A, then B): its sealed test "
                             "is E3's headline number. Nothing switches." % chosen)
            for k in q:
                lines.append("A person reads E3-%s's sealed test once: python -m weed_optimizer_framework.tools.inc2."
                             "twostage test-read --arm %s, submit the printed sbatch argv, then python -m "
                             "weed_optimizer_framework.tools.inc2.twostage test-report --arm %s (reported against "
                             "b_v2_m640's 0.8786, with the gap to 0.90)." % (k, k, k))
        else:
            lines.append("No arm qualifies: no test is read; nothing switches.")
            title = "E3's verdict: no arm qualifies"
        self._card("research", title, "\n".join(lines), trigger=["L23G"])
        st["e3_card"] = self.utc

    # ------------------------------------------------------------------ V3F (2026-10-11)
    def _v3fam(self):
        """The family's own record in the stream state: items {key: {state, fails, estimate, spend, settled,
        proposal, exp}} (key data, score:<arm>, build:<arm>, detr:smoke|train, verdict) and the cards raised."""
        f = self.st.setdefault("v3fam", {})
        f.setdefault("items", {})
        return f

    @staticmethod
    def _v3fam_key(lever, params):
        return {"L23H": "data", "L23L": "verdict"}.get(lever) or {
            "L23J": "score:%s", "L23I": "build:%s", "L23K": "detr:%s"}[lever] % (
            params.get("verb") if lever == "L23K" else params.get("arm"))

    def _v3fam_item(self, lever, params):
        return self._v3fam()["items"].setdefault(self._v3fam_key(lever, params), {})

    def _v3fam_amendment(self):
        """True when the deployed tree (the lab's, which the S23 check keeps equal to the cluster's) holds the
        amendment's file and its Revisions' (29-31) with the sha256s the domain records: the deployed commit
        descends from both."""
        fam = (self.dom or {}).get("v3fam") or {}
        if not fam.get("amendment_file"):
            return False
        for f, sha in ((fam["amendment_file"], fam.get("amendment_sha256")),
                       (fam.get("revision_file"), fam.get("revision_sha256"))):
            if not f:
                continue
            try:
                if hashlib.sha256((self.paths.lab_inc.parents[3] / f).read_bytes()).hexdigest() != sha:
                    return False
            except OSError:
                return False
        return True

    def _v3fam_deployed(self, f, sha):
        """True when the deployed tree (the lab's, as _v3fam_amendment reads it) holds file f with this sha256."""
        if not f or not sha:
            return False
        try:
            return hashlib.sha256((self.paths.lab_inc.parents[3] / f).read_bytes()).hexdigest() == sha
        except OSError:
            return False

    def _v3fam_reopen_blocks(self):
        """The domain's reopen blocks in order: v3fam.reopen (deviation 4's, an object, kept so for a ticker of an
        earlier deploy that still reads it during a deploy), then v3fam.reopens (deviation 5's and later, a list)."""
        from . import stream_remote as SR
        return SR.v3fam_reopen_blocks(self.dom)

    def _v3fam_reopen(self):
        """A recorded deviation's reopening (deviation 4, 2026-10-20: the data build, after its defect is fixed, and
        the verdict, whose record of 2026-10-13 was decided while seven arms had never started and stays as it is;
        deviation 5, 2026-10-24: the score jobs of D8, D4F, D4N, D4R15, D1 and D5B, which the platform's own stale
        reproduction records refused on 2026-10-21, and the verdict, whose continuation of 2026-10-22 recorded those
        arms failed and stays as it is): each item a block of the domain's v3fam.reopen names is proposed again once
        the deployed tree holds that deviation's file with its sha256, once per marker (an item whose history holds
        the marker is never reopened by it again), and only an item whose last attempt started before the deviation
        (the block's before_utc): an item run after it is never reopened by it. Such an item recorded failed or
        done has its spend so far moved into prior_gpuh (every attempt stays counted against the family cap), its
        earlier proposal ids added to failed_ids (a new proposal takes a new id: the executor runs an id once), its
        record kept under reopen_history, and its state set to missing; the verdict's card mark goes with it (the
        next verdict raises its own card). An item running or never proposed is left as it is. Returns the keys
        reopened now."""
        out = []
        for ro in self._v3fam_reopen_blocks():
            out += self._v3fam_reopen_block(ro)
        return out

    def _v3fam_reopen_block(self, ro):
        marker = ro.get("marker")
        if not marker or not ro.get("items") or not self._v3fam_deployed(ro.get("file"), ro.get("sha256")):
            return []
        f = self._v3fam()
        out = []
        for key in ro["items"]:
            rec = f["items"].get(key)
            if not isinstance(rec, dict) or rec.get("state") not in ("failed", "done", DS.V3FAM_HELD_PLATFORM) \
                    or rec.get("reopened") == marker \
                    or marker in [h.get("marker") for h in rec.get("reopen_history") or [] if isinstance(h, dict)]:
                continue
            if not ro.get("before_utc") or not rec.get("started_utc") or str(rec["started_utc"]) >= ro["before_utc"]:
                continue
            before = {k: v for k, v in rec.items() if k != "reopen_history"}
            ids = [a.get("proposal") for a in rec.get("attempts") or []] + [rec.get("proposal")]
            self._v3fam_fold(rec)
            hist = list(rec.get("reopen_history") or []) + [dict(before, marker=marker, reopened_utc=self.utc)]
            if key == "verdict" and f.get("verdict_card"):
                hist[-1]["verdict_card"] = f.pop("verdict_card")
            rec.update(state="missing", fails=0, proposal=None, reopened=marker, reopened_utc=self.utc,
                       reopen_history=hist)
            rec.pop("why", None)
            rec.pop("platform_refusals", None)
            rec.pop("held_for_person", None)
            failed = self._failed_ids()
            self.st["failed_ids"] = (failed + [i for i in ids if i and i not in failed])[-FAILED_IDS_KEEP:]
            out.append(key)
        if out:
            self._ledger("v3fam_reopened", record_only=True, v3fam_items=out, marker=marker, reasons=[
                "V3F: %s reopened under %s (%s): proposed again under new ids, earlier spend kept in prior_gpuh"
                % (", ".join(out), marker, ro.get("file"))])
            self._card("research", "V3F: %s proposed again (%s)" % (", ".join(out), marker),
                       "%s, recorded in %s. The family's verdicts already written stay recorded as they are; the next "
                       "verdict comes once every arm's record is complete or failed (V3F.7 step 11). Earlier attempts "
                       "stay counted against the family cap." % (ro.get("why") or marker, ro.get("file")),
                       lever="L23L")
        return out

    def _v3fam_stage(self, exp_status, payload=None):
        """/stage/v3fam (V3F.6.4), what DR0's family block and the cards read: the gate's inputs (the amendment
        deployed, the time stamp, D0-S's record and trigger, the EXIF pre-flight), every item's state (done from
        its own record where one exists: the data summary, an arm's score record (complete, or 'not run' with its
        reason), an experiment done, D6's smoke and seeds, the verdict decided; else what the platform ran:
        running, failed_once, failed, not run (family cap); blocked and not needed from the domain and the
        trigger; an arm whose dependency ended without its effect is not run), and the family's GPU-h against
        its cap. Deviation 5 (2026-10-24): the reproduction records against the deployed code (equivalence: current;
        stale while a record is taken by other code, or, once the score that takes them is done, one the domain names
        (v3fam.equivalence_records) is not in place; unknown
        while the snapshot does not report them; a record a recorded deviation supersedes by sha256 is neither), an
        item held after a refusal of the platform's own making released once they are current, and the first
        snapshot that shows superseded records moved aside written to the ledger. None without a family block."""
        from . import stream_remote as SR
        fam = (self.dom or {}).get("v3fam") or {}
        if not fam:
            return None
        self._v3fam_reopen()
        items = self._v3fam()["items"]
        status = self._artifact(fam.get("status") or SR.V3FAM_STATUS) or {}

        def mine(key):
            return (items.get(key) or {}).get("state")
        sw = status.get("sweep") or {}
        out = {"amendment": self._v3fam_amendment(),
               "timestamp": bool((status.get("timestamp") or {}).get("exists")), "sweep": bool(sw.get("complete")),
               "exif": bool((status.get("exif") or {}).get("passed")),
               "trigger": {k: bool(sw.get(k)) for k in ("bclean", "dev_branch", "test_branch")}}
        summ = self._artifact(fam.get("data_record") or "splits/v3fam/summary.json") or {}
        data_ok = summ.get("status") == "complete" or bool((status.get("data") or {}).get("complete"))
        out["data"] = "done" if data_ok else (mine("data") or "missing")
        b = {}
        for arm in fam.get("builds") or []:
            x = exp_status.get((fam.get("build_exps") or {}).get(arm))
            own = mine("build:%s" % arm)
            if arm in (fam.get("blocked_builds") or {}):
                b[arm] = "blocked"
            elif arm == "D0BCLEAN" and not out["trigger"]["bclean"]:
                b[arm] = "not needed"
            elif x == "done":
                b[arm] = "done"
            elif own in ("failed", V3FAM_NOT_RUN_CAP):
                b[arm] = own
            elif x == "built":
                b[arm] = "built"
            elif own == "done":
                b[arm] = "running"               # its build job ended; the experiment shows in the next snapshot
            else:
                b[arm] = own or "missing"
        out["build"] = b
        det = status.get("detr") or {}
        sm = det.get("smoke")
        if sm == "complete" and det.get("walltime_ok") is True:
            smoke = "done"
        elif sm and str(sm).startswith("not run"):
            smoke = str(sm)
        else:
            smoke = mine("detr:smoke") or "missing"
            smoke = "running" if smoke == "done" else smoke     # its job ended; its record shows next snapshot
        if sorted(int(k) for k in det.get("trained") or []) == [0, 1, 2]:
            train = "done"
        elif DS.v3fam_terminal(smoke) and smoke != "done":
            train = smoke
        else:
            train = mine("detr:train") or "missing"
            train = "running" if train == "done" else train
        out["detr"] = {"smoke": smoke, "train": train, "smoke_sha256": det.get("smoke_sha256"),
                       "epoch_seconds": det.get("epoch_seconds"), "projection_h": det.get("projection_h")}
        if out["data"] == "failed":
            # the data build ended without its summary after its second failure: nothing that trains on the family's
            # manifests (the builds) or stages after it (D6) can run; recorded so for the family table, but these
            # items never started, so the verdict does not take them as ended (DS.v3fam_verdict_ready; deviation 4)
            why = DS.V3FAM_NOT_RUN_DATA
            for arm in b:
                if b[arm] in ("missing", "failed_once"):
                    b[arm] = why
            for k in ("smoke", "train"):
                if out["detr"][k] in ("missing", "failed_once"):
                    out["detr"][k] = why
        sc = {}
        arms = list(fam.get("pre_build_scores") or []) + list(fam.get("post_build_scores") or [])
        for arm in arms:
            rec = self._artifact((fam.get("score_record") or "capacity/v3fam_score_%s.json") % arm.lower()) or {}
            if rec.get("status") == "complete":
                sc[arm] = "done"
            elif rec.get("status") == "not run":
                sc[arm] = "not run (%s)" % (rec.get("reason") or "recorded")
            else:
                own = mine("score:%s" % arm)
                sc[arm] = "running" if own == "done" else (own or "missing")
        out["score"] = sc
        for arm in arms:
            if sc[arm] in ("missing", "failed_once") \
                    and DS._v3fam_deps(out, (fam.get("score_after") or {}).get(arm)) == "not run":
                sc[arm] = "not needed" if arm == "BCLEAN" and b.get("D0BCLEAN") == "not needed" else \
                    "not run (a dependency ended without its effect)"
        vr = self._artifact(fam.get("verdict_record") or "capacity/v3fam_v1.json") or {}
        out["verdict"] = "done" if vr.get("status") == "decided" else (
            "running" if mine("verdict") == "done" else (mine("verdict") or "missing"))
        out["equivalence"] = self._v3fam_equivalence(status, sc)
        self._v3fam_release(out)
        out["spend_gpuh"] = round(self._v3fam_spend(exp_status, status, payload), 4)
        out["cap_gpuh"] = float(fam.get("cap_gpu_hours") or 260.0)
        out["cap_refused"] = sorted(k for k, v in items.items() if v.get("state") == V3FAM_NOT_RUN_CAP)
        out["held_person"] = sorted(k for k, v in items.items()
                                    if v.get("state") == DS.V3FAM_HELD_PLATFORM and v.get("held_for_person"))
        return out

    def _v3fam_equivalence(self, status, sc):
        """/stage/v3fam/equivalence from the snapshot's block (stream_remote.v3fam_equivalence): {state: current |
        stale | unknown, stale, missing, fingerprint, superseded, supersede}. Stale: a record taken by other code
        than the deployed one (inc2.v3fam.require_equivalence refuses every family score resting on it), or, once the
        score item that takes them is done, a record the domain names (v3fam.equivalence_records) not in place. The
        first snapshot that shows records a
        recorded deviation supersedes moved aside writes one ledger event."""
        eq = status.get("equivalence")
        if not isinstance(eq, dict) or not isinstance(eq.get("code"), dict):
            return {"state": "unknown", "why": _short(status.get("equivalence_error"), 200)
                    or "the snapshot does not report the reproduction records"}
        present = {r.get("name") for r in eq.get("records") or [] if isinstance(r, dict)}
        # the records the domain names (v3fam.equivalence_records: those E3RS's job takes, V3F.3.2) must be in place
        # once the score item that takes them is done
        er = ((self.dom or {}).get("v3fam") or {}).get("equivalence_records") or {}
        missing = sorted(set(er.get("names") or ()) - present) if er.get("after") and sc.get(er["after"]) == "done" \
            else []
        stale = sorted(str(x) for x in eq.get("stale") or [])
        sup = eq.get("supersede") or {}
        moved = sorted(str(x) for x in sup.get("moved") or [])
        f = self._v3fam()
        if moved and f.get("superseded_seen") != moved:
            f["superseded_seen"] = moved
            self._ledger("v3fam_superseded", record_only=True, records=moved, reasons=[
                "the family records a recorded deviation supersedes are moved aside byte for byte (each in its dated "
                "folder's MANIFEST.json with its sha256): %s" % ", ".join(moved)])
        return {"state": "stale" if (stale or missing) else "current", "stale": stale, "missing": missing,
                "fingerprint": eq.get("fingerprint"), "superseded": sorted(eq.get("superseded") or []),
                "supersede": {k: sorted(sup.get(k) or []) for k in ("pending", "moved", "other")}}

    def _v3fam_release(self, out):
        """An item held after a refusal of the platform's own making (DS.V3FAM_HELD_PLATFORM) is released (missing:
        DR0 proposes it again under a new id) once the snapshot reports the reproduction records current, unless it
        was refused V3FAM_PLATFORM_MAX times or by a deterministic check (held_for_person: deviation 6's loader check,
        E3's ground-truth and frame checks); then it stays held for a person. The stage shows it as released."""
        if (out.get("equivalence") or {}).get("state") != "current":
            return
        items = self._v3fam()["items"]
        rel = []
        for key, rec in sorted(items.items()):
            if rec.get("state") == DS.V3FAM_HELD_PLATFORM and int(rec.get("platform_refusals") or 0) < \
                    V3FAM_PLATFORM_MAX and not rec.get("held_for_person"):
                rec.update(state="missing", released_utc=self.utc)
                rel.append(key)
                if key.startswith("score:") and (out.get("score") or {}).get(key[6:]) == DS.V3FAM_HELD_PLATFORM:
                    out["score"][key[6:]] = "missing"
        if rel:
            self._ledger("v3fam_released", record_only=True, v3fam_items=rel, reasons=[
                "the reproduction records are current under the deployed code: %s proposed again under new ids"
                % ", ".join(rel)])

    def _v3fam_report(self, exp, payload):
        """An experiment's report.json from this snapshot, else from its frozen record."""
        snap = (((payload or {}).get("experiments") or {}).get(exp) or {}).get("snapshot") or {}
        rep = ((snap.get("decision") or {}).get("artifacts") or {}).get("%s/report.json" % exp)
        if not isinstance(rep, dict):
            fz = _read_json(self.paths.frozen(self.name, exp)) or {}
            rep = (((fz.get("snapshot") or {}).get("decision") or {}).get("artifacts") or {}).get(
                "%s/report.json" % exp)
        return rep if isinstance(rep, dict) else None

    def _v3fam_spend(self, exp_status, status, payload=None):
        """The family's GPU-h (decision 25): each item the platform ran at its estimate until it settled (a job
        from sacct, a build from its experiment's report once done), plus what its earlier attempts settled
        (prior_gpuh: a retry adds to the first, V3F.6.4 and V3F.7), a refused item nothing more; the person steps
        (the sweep, the reads, the re-analysis) from the derived status, a running one at its estimate."""
        fam = (self.dom or {}).get("v3fam") or {}
        total = 0.0
        for key, it in self._v3fam()["items"].items():
            total += float(it.get("prior_gpuh") or 0.0)       # the item's earlier attempts (_v3fam_fold)
            if it.get("state") == V3FAM_NOT_RUN_CAP or it.get("spend") is None:
                continue
            if key.startswith("build:") and it.get("spend") == "open" and exp_status.get(it.get("exp")) == "done":
                rep = self._v3fam_report(it["exp"], payload)
                gh = (rep or {}).get("gpu_hours") if isinstance(rep, dict) and rep.get("done") else None
                if isinstance(gh, dict):
                    it["settled"] = round(sum(float((x or {}).get("hours") or 0.0) for x in gh.values()
                                              if isinstance(x, dict)), 4)
                    it["spend"] = "settled"
            total += float(it["settled"]) if it.get("spend") == "settled" and it.get("settled") is not None \
                else float(it.get("estimate") or 0.0)
        ph = ((fam.get("costs") or {}).get("person_hours") or {})
        total += float(status.get("person_settled_gpuh") or 0.0)
        for j in status.get("person_jobs") or []:
            if j.get("state") == "running":
                n = str(j.get("job") or "")
                total += float(ph.get("sweep") if n.startswith("v3fam_sweep") else
                               ph.get("retest") if n.startswith("v3fam_retest") else ph.get("read") or 0.0)
        return total

    def _v3fam_params(self, lever, pr, pkg, info):
        """(params, child_exp) of a family lever from DR0's item, or None."""
        fam = (self.dom or {}).get("v3fam") or {}
        if lever == "L23H":
            return {"pkg": pkg}, None
        if lever == "L23L":
            # the items this ticker recorded 'not run (family cap)': the verdict writes them under v3fam/v1/cap so
            # the family table gives that reason (V3F.3.4, V3F.3.5); and those it recorded failed after their own
            # second failure (a score, build or D6 job; never the data build, whose failure ends nothing): the verb
            # counts the arms they end as ended, as DS.v3fam_verdict_ready does, although such a job may have failed
            # before it wrote a provenance attempt (deviation 4)
            items = self._v3fam()["items"]
            refused = sorted(k for k, v in items.items() if v.get("state") == V3FAM_NOT_RUN_CAP and k != "verdict")
            failed = sorted(k for k, v in items.items() if v.get("state") == "failed" and k not in ("verdict", "data"))
            return dict({"pkg": pkg}, **({"cap_refused": ",".join(refused)} if refused else {}),
                        **({"ticker_failed": ",".join(failed)} if failed else {})), None
        if lever == "L23J":
            arms = list(fam.get("pre_build_scores") or []) + list(fam.get("post_build_scores") or [])
            if pr.get("arm") not in arms:
                return None
            info["arm"] = pr["arm"]
            return {"pkg": pkg, "arm": pr["arm"]}, None
        if lever == "L23K":
            if pr.get("verb") not in ("smoke", "train"):
                return None
            if pr["verb"] == "smoke":
                return {"pkg": pkg, "verb": "smoke"}, None
            # the seeds not yet trained (a retry never resubmits a trained seed, which refuses a second training)
            from . import stream_remote as SR
            st_ = self._artifact(fam.get("status") or SR.V3FAM_STATUS) or {}
            done = {int(k) for k in ((st_.get("detr") or {}).get("trained") or [])}
            seeds = [x for x in (fam.get("seeds") or [0, 1, 2]) if int(x) not in done]
            if not seeds:
                return None
            return {"pkg": pkg, "verb": "train", "seeds": ",".join(str(int(x)) for x in seeds)}, None
        arm = pr.get("arm")
        exp = (fam.get("build_exps") or {}).get(arm)
        if not exp or arm in (fam.get("blocked_builds") or {}):
            return None
        from . import stream_remote as SR
        st_ = self._artifact(fam.get("status") or SR.V3FAM_STATUS) or {}
        a = ((st_.get("data") or {}).get("arms") or {}).get(arm) or \
            ((self._artifact(fam.get("data_record") or "splits/v3fam/summary.json") or {}).get("arms") or {}).get(arm) \
            or {}
        if arm in ("D8M", "D8S"):
            a = dict(a, images=a.get("images") or 3048)          # D8 trains LOCK v2's train_core (3,048 rows)
        if not a.get("images"):
            return None
        info.update(images=int(a["images"]), arm=arm)
        return {"pkg": pkg, "exp": exp, "arm": arm, "seeds": "0,1,2", "role": "baseline"}, exp

    def _v3fam_cap_ok(self, lever, params, est):
        """False (recorded 'not run (family cap)', one card, never retried) when est would take the family past
        its cap: the family's GPU-h as this tick's context gives it, plus the estimates proposed this tick."""
        fam = (self.dom or {}).get("v3fam") or {}
        cap = float(fam.get("cap_gpu_hours") or 260.0)
        ctx = (self.ev.json(E.CONTEXT) or {}) if self.ev is not None else {}
        spent = float((((ctx.get("stage") or {}).get("v3fam") or {}).get("spend_gpuh")) or 0.0)
        tick = getattr(self, "_v3fam_tick", None)
        if tick is None or tick[0] != self.utc:
            tick = self._v3fam_tick = [self.utc, 0.0]
        after = spent + tick[1] + float(est or 0.0)
        if after <= cap + V3FAM_CAP_EPS:
            tick[1] += float(est or 0.0)
            return True
        rec = self._v3fam_item(lever, params)
        if rec.get("state") != V3FAM_NOT_RUN_CAP:
            key = self._v3fam_key(lever, params)
            self._v3fam_fold(rec)                 # what an earlier attempt of it used stays counted
            rec.update(state=V3FAM_NOT_RUN_CAP, refused_utc=self.utc, spent_gpuh=round(spent + tick[1], 4),
                       estimate=float(est or 0.0), spend=None)
            self._ledger("not_taken", lever=lever, v3fam_item=key, reasons=[
                "%s: %s (%.2f GPU-h) would take the family from %.2f past its cap of %.0f GPU-h (decision 25)"
                % (V3FAM_NOT_RUN_CAP, key, float(est or 0.0), spent + tick[1], cap)])
            self._card("escalation", "V3F's %s: %s" % (key, V3FAM_NOT_RUN_CAP),
                       "%s (%.2f GPU-h est.) would take the v3 family from %.2f GPU-h (running items at their "
                       "estimates, ended ones settled) past its cap of %.0f GPU-h (decision 25). It is recorded "
                       "'%s', keeps its place in the family table, in C and in its Holm set, and is never retried."
                       % (key, float(est or 0.0), spent + tick[1], cap, V3FAM_NOT_RUN_CAP), lever=lever)
        return False

    @staticmethod
    def _v3fam_fold(rec):
        """Move an item's spend so far into prior_gpuh (an attempt that ended, before the item is proposed again or
        refused at the cap): what sacct settled, or, for an attempt never settled, its estimate (overstated, never
        understated). The cap counts every attempt (V3F.6.4, V3F.7)."""
        got = None
        if rec.get("spend") == "settled" and rec.get("settled") is not None:
            got = float(rec["settled"])
        elif rec.get("spend") == "open":
            got = float(rec.get("estimate") or 0.0)
        if got is not None:
            rec["prior_gpuh"] = round(float(rec.get("prior_gpuh") or 0.0) + got, 4)
            rec.setdefault("attempts", []).append({"proposal": rec.get("proposal"), "gpuh": round(got, 4),
                                                   "how": rec.get("spend")})
        rec.update(spend=None, settled=None)

    def _v3fam_started(self, it):
        p = it["proposal"]
        rec = self._v3fam_item(it["lever"], p.get("params") or {})
        if rec.get("proposal") == p["id"] and rec.get("spend") in ("open", "settled"):
            # the same run taken back (adopted after a restart): its spend stands, it is not a new attempt
            rec.update(state="running", lever=it["lever"], exp=p.get("child_exp"))
            return
        self._v3fam_fold(rec)
        rec.update(state="running", estimate=float(p.get("est_gpu_hours") or 0.0), spend="open", settled=None,
                   proposal=p["id"], lever=it["lever"], started_utc=self.utc, exp=p.get("child_exp"))

    def _v3fam_settle(self, it, rows, build_ended=False):
        """A family job's spend settled from sacct once every job of it ended (V100 GPU-h). A build (L23I) is
        settled by its experiment's report, unless its job ended without the experiment (build_ended): then by
        its build job's sacct."""
        if not rows or not all(r and r.get("elapsed_s") is not None for r in rows):
            return
        rec = self._v3fam_item(it["lever"], it["proposal"].get("params") or {})
        if it["lever"] == "L23I" and not build_ended:
            return                                # its runs are settled by the experiment's report
        rec["settled"] = round(sum(float(r["elapsed_s"]) * float(r.get("gpu_count") or 1) / 3600.0 for r in rows), 4)
        rec["spend"] = "settled"

    def _v3fam_failed(self, ln, it, why, charged, ended=None):
        """A family item that ended without its effect: one card; proposed once more under a new id; after its
        second failure (v3fam.max_failures) recorded failed, its id declined, the family's other items go on. No
        lane failure, no stop-loss. An L23K seed job that ended TIMEOUT (its 36 h walltime) fails D6 at once and is
        never retried (V3F.7: killed before its export, D6's scoring and read are not spent). A build that ended
        without its experiment (ended) is charged its build job's sacct (settled by the follow), else the
        domain's build-job hours; any other job its estimate when charged."""
        st = self.st
        p = it["proposal"]
        lever, params = it["lever"], p.get("params") or {}
        if V3FAM_PLATFORM_REFUSAL_RE.search(str(why or "")):
            return self._v3fam_platform_refusal(ln, it, why, charged)
        fam = (self.dom or {}).get("v3fam") or {}
        key = self._v3fam_key(lever, params)
        rec = self._v3fam_item(lever, params)
        rec["fails"] = int(rec.get("fails") or 0) + 1
        if rec.get("spend") == "open":
            if ended is not None and lever == "L23I":
                rec["settled"] = float(((fam.get("costs") or {}).get("build_job_hours")) or 0.0)
            else:
                rec["settled"] = float(rec.get("estimate") or 0.0) if charged else 0.0
            rec["spend"] = "settled"
        walltime = lever == "L23K" and params.get("verb") == "train" and any(
            str(x).startswith("TIMEOUT") for x in it.get("v3fam_states") or [])
        last = walltime or rec["fails"] >= int(fam.get("max_failures") or 2)
        rec["state"] = "failed" if last else "failed_once"
        if walltime:
            rec["why"] = "walltime: a seed job was killed at its 36 h limit before its export (V3F.7)"
        if last:
            st["declined"] = (list(st.get("declined") or []) + [p["id"]])[-200:]
        else:
            st["failed_ids"] = (self._failed_ids() + [p["id"]])[-FAILED_IDS_KEEP:]
        self._ledger("failed", lane=ln, lever=lever, reasons=[_short(why, 1000)], proposal_id=p["id"],
                     charged=charged, record_only=True, v3fam_item=key, fails=rec["fails"], walltime=walltime)
        title = RECORD_ONLY_TITLES[lever]
        title = title % (params.get("arm") or params.get("verb")) if "%s" in title else title
        if walltime:
            body = ("%s. A seed killed at its walltime fails D6 (V3F.7; L23K's falsifier): recorded failed, never "
                    "retried; D6's scoring and its reference read are not spent, P5 and D6's Tier C arm are 'not "
                    "run' and keep their places (V3F.3.4). The family's other items go on." % _short(why, 400))
        elif last and lever == "L23H":
            body = ("%s. The family's data build failed twice: the arms it builds (and D6, which stages after it) "
                    "never start, so the verdict waits until every arm's record is complete or failed (V3F.7 step "
                    "11; deviation 4, which records why this wait departs from V3F.3.4/V3F.3.5's 'a job that failed "
                    "twice'). The family's other items go on. A person finds the cause, fixes it as a recorded "
                    "deviation and names the build in the domain's v3fam.reopen under a new marker (its before_utc "
                    "after this failure); the platform then proposes it again. To rerun by hand: %s"
                    % (_short(why, 400), self._v3fam_rerun(lever, params)))
        else:
            body = "%s. Record only: %s; the family's other items go on. A person may rerun it from the login node: %s" \
                % (_short(why, 400), "after its second failure it is recorded failed and keeps its place in the "
                                     "family's verdict as failed" if last else
                   "it is proposed once more under a new id", self._v3fam_rerun(lever, params))
        self._card("escalation", "%s%s" % (title, ": recorded failed" if last else ""), body,
                   lever=lever, trigger=list(p.get("trigger") or []))
        self._clear(ln)

    # ------------------------------------------------------------------ DS1 (2026-10-29)
    def _ds1(self):
        """DS1's own record in the stream state: items {key: {state, fails, estimate, spend, settled, proposal, phase,
        gb}} (keys data:<phase>:<step>, sheets:<kind>:<batch>, build:<arm>, long:<arm>:<verb>, score:<arm>[:<draft>],
        verdict:<phase>), the bytes phase 1 wrote (estimated), the cards raised."""
        f = self.st.setdefault("ds1", {})
        f.setdefault("items", {})
        f.setdefault("cards", {})
        return f

    @staticmethod
    def _ds1_key(lever, params):
        p = params or {}
        if lever == "L30":
            return "data:%s:%s" % (p.get("phase"), p.get("step"))
        if lever == "L31":
            return "sheets:%s:%s" % (p.get("kind"), p.get("batch"))
        if lever == "L32":
            return "build:%s" % p.get("arm")
        if lever == "L32X":
            return "long:%s:%s" % (p.get("arm"), p.get("verb"))
        if lever == "L33":
            return "score:%s" % p.get("arm")
        return "verdict:%s" % p.get("phase")

    def _ds1_item(self, it):
        p = it["proposal"]
        key = p.get("ds1_item") or self._ds1_key(it["lever"], p.get("params") or {})
        return key, self._ds1()["items"].setdefault(key, {})

    def _ds1_tree_file(self, rel, sha=None, marker=None):
        """True when the deployed tree (the lab's, which the S23 check keeps equal to the cluster's) holds `rel` (a
        git-root path, or a package path under weed_llm_benchmark/) with this sha256 or this marker in it."""
        for base in (self.paths.lab_inc.parents[3], self.paths.lab_inc.parents[2]):
            p = base / rel
            try:
                data = p.read_bytes()
            except OSError:
                continue
            if sha is not None:
                return hashlib.sha256(data).hexdigest() == sha
            return marker is None or marker.encode("utf-8") in data
        return False

    def _ds1_reopen(self):
        """A DS1 Revision's reopening (Revision 2, 2026-10-29: the power pre-flight score:POWER:<draft>, refused twice
        by a defect of the DS1 pass's stage-1 check and recorded failed, which held MD-S0's freeze and every training
        item): each item a block of the domain's ds1.reopens names is proposed again once the deployed tree holds that
        Revision's file with its sha256, once per marker, and only if its last attempt started before the block's
        before_utc and it is recorded failed. Its spend so far moves into prior_gpuh (every attempt stays counted
        against the family cap), its earlier proposal ids into failed_ids (a new proposal takes a new id: the executor
        runs an id once), its record into reopen_history, its state to missing. A block's "states" may also name
        "missing" (DS1 deviation 2: an L30 data step goes back to missing when it is done, and runs again when the
        cluster's status calls for it): such an item is reopened the same way, its spend kept, so the run the deviation
        calls for is its own attempt on record. Returns the keys reopened now."""
        cfg = (self.dom or {}).get("ds1") or {}
        f = self._ds1()
        out = []
        for ro in [b for b in cfg.get("reopens") or [] if isinstance(b, dict)]:
            marker = ro.get("marker")
            if not marker or not ro.get("items") or not self._ds1_tree_file(ro.get("file") or "",
                                                                            sha=ro.get("sha256")):
                continue
            states = [x for x in (ro.get("states") or ["failed"]) if x in ("failed", "missing")]
            now = []
            for key in ro["items"]:
                rec = f["items"].get(key)
                if not isinstance(rec, dict) or rec.get("state") not in states or rec.get("reopened") == marker or \
                        marker in [h.get("marker") for h in rec.get("reopen_history") or [] if isinstance(h, dict)]:
                    continue
                if not ro.get("before_utc") or not rec.get("started_utc") or \
                        str(rec["started_utc"]) >= str(ro["before_utc"]):
                    continue
                before = {k: v for k, v in rec.items() if k != "reopen_history"}
                ids = [a.get("proposal") for a in rec.get("attempts") or [] if isinstance(a, dict)] + \
                    [rec.get("proposal")]
                self._v3fam_fold(rec)
                rec.update(state="missing", fails=0, proposal=None, reopened=marker, reopened_utc=self.utc,
                           reopen_history=list(rec.get("reopen_history") or []) + [
                               dict(before, marker=marker, reopened_utc=self.utc)])
                rec.pop("why", None)
                failed = self._failed_ids()
                earlier = [i for i in ids if i and i not in failed]
                self.st["failed_ids"] = (failed + earlier)[-FAILED_IDS_KEEP:]
                now.append(key)
            if now:
                self._ledger("ds1_reopened", record_only=True, ds1_items=now, marker=marker, reasons=[
                    "DS1: %s reopened under %s (%s): proposed again under a new id once DS1's gate is open, earlier "
                    "spend kept in prior_gpuh" % (", ".join(now), marker, ro.get("file"))])
                self._card("research", "DS1: %s reopened (%s)" % (", ".join(now), marker),
                           "%s, recorded in %s. It is proposed again under a new id once DS1's gate is open (every "
                           "time-stamp record in place). Earlier attempts stay counted against the family cap." % (
                               ro.get("why") or marker, ro.get("file")),
                           lever="L30" if all(k.startswith("data:") for k in now) else "L33")
                out += now
        return out

    def _ds1_stage(self, exp_status, payload=None):
        """/stage/ds1 (DS.9.3), what DR0's DS1 block and DS1's cards read: the gate's inputs (the amendment, its
        Revisions and recorded deviations and the cutter's never-cut change in the deployed tree, the time stamp), the
        cluster's derived status (stream_remote.ds1_status), every item's state (done from its own record where one
        exists: an experiment done, a score record complete, the long path's seeds trained, a verdict decided; else
        what the platform ran), the family's GPU-h against its cap, the bytes phase 1 wrote. None without a ds1
        block."""
        from . import stream_remote as SR
        cfg = (self.dom or {}).get("ds1") or {}
        if not cfg:
            return None
        self._ds1_reopen()
        f = self._ds1()
        status = self._artifact(cfg.get("status") or SR.DS1_STATUS) or {}
        nc = cfg.get("never_cut") or {}
        items = {k: v.get("state") for k, v in f["items"].items()}
        blocked = f.get("runs_blocked") or {}
        for arm, exp in (cfg.get("build_exps") or {}).items():
            x = exp_status.get(exp)
            own = items.get("build:%s" % arm)
            if x == "done":
                items["build:%s" % arm] = "done"
            elif own == "done":
                # its build job ended; the runs go on, unless they are blocked or stale (a person's card): then the
                # arm is failed for DS1's plan and verdict until it advances again (DS.9.3, V3F.6.4)
                items["build:%s" % arm] = "failed" if exp in blocked else "running"
        for arm, stt in (status.get("scores") or {}).items():
            if stt == "complete":
                items["score:%s" % arm] = "done"
        for arm, ls in ((status.get("long") or {}).get("arms") or {}).items():
            sm = ls.get("smoke")
            if sm == "complete":
                items["long:%s:smoke" % arm] = "done"
            elif sm and str(sm).startswith("not run"):
                items["long:%s:smoke" % arm] = str(sm)
                items["long:%s:train" % arm] = str(sm)
            if sorted(int(k) for k in ls.get("trained") or []) == [0, 1, 2]:
                items["long:%s:train" % arm] = "done"
        for ph, v in (status.get("verdicts") or {}).items():
            if (v or {}).get("status") == "decided":
                items["verdict:%s" % ph] = "done"
        # the amendment, each Revision and each recorded deviation (DS1 deviation 1, 2026-10-30: the domain's
        # ds1.deviations) in the deployed tree, each by the sha256 the domain pins
        out = {"amendment": self._ds1_tree_file(cfg.get("amendment_file") or "", sha=cfg.get("amendment_sha256"))
               and all(self._ds1_tree_file(r.get("file") or "", sha=r.get("sha256"))
                       for r in list(cfg.get("revisions") or []) + list(cfg.get("deviations") or [])),
               "never_cut": self._ds1_tree_file(nc.get("module") or "", marker=nc.get("marker")),
               "timestamp": bool(status.get("timestamp")), "status": status, "items": items,
               "today": str(self.utc)[:10],
               "phase1_gb": round(float(f.get("phase1_gb") or 0.0), 3),
               "phase1_charged": sorted(f.get("phase1_charged") or []),
               "audit_render_for": f.get("audit_render_for"),
               "spend_gpuh": round(self._ds1_spend(exp_status, status, payload), 4),
               "cap_gpuh": float(cfg.get("cap_gpu_hours") or 260.0),
               "cap_refused": sorted(k for k, v in f["items"].items() if v.get("state") == DS1_NOT_RUN_CAP)}
        return out

    def _ds1_spend(self, exp_status, status, payload=None):
        """DS1's GPU-h (DS.0 item 10, as V3F decision 25): each item at its estimate until it settled (a job from
        sacct, a build from its experiment's report), plus what its earlier attempts settled; the person jobs (the
        sealed reads) from the derived status, a running one at its estimate."""
        cfg = (self.dom or {}).get("ds1") or {}
        total = 0.0
        for key, it in self._ds1()["items"].items():
            total += float(it.get("prior_gpuh") or 0.0)
            if it.get("state") == DS1_NOT_RUN_CAP or it.get("spend") is None:
                continue
            if key.startswith("build:") and it.get("spend") == "open" and exp_status.get(it.get("exp")) == "done":
                rep = self._v3fam_report(it["exp"], payload)
                gh = (rep or {}).get("gpu_hours") if isinstance(rep, dict) and rep.get("done") else None
                if isinstance(gh, dict):
                    it["settled"] = round(float(it.get("build_job_gpuh") or 0.0) + sum(
                        float((x or {}).get("hours") or 0.0) for x in gh.values() if isinstance(x, dict)), 4)
                    it["spend"] = "settled"
            total += float(it["settled"]) if it.get("spend") == "settled" and it.get("settled") is not None \
                else float(it.get("estimate") or 0.0)
        ph = ((cfg.get("costs") or {}).get("person_hours") or {})
        total += float(status.get("person_settled_gpuh") or 0.0)
        for j in status.get("person_jobs") or []:
            if j.get("state") == "running":
                total += float(ph.get("read") or 0.0)
        return total

    def _ds1_params(self, lever, pr, pkg, info):
        """(params, child_exp) of a DS1 lever from DR0's item, or None."""
        cfg = (self.dom or {}).get("ds1") or {}
        st_ = (((self.ev.json(E.CONTEXT) or {}).get("stage") or {}).get("ds1") or {}).get("status") \
            if self.ev is not None else None
        if not isinstance(st_, dict):
            from . import stream_remote as SR
            st_ = self._artifact(cfg.get("status") or SR.DS1_STATUS) or {}
        if lever == "L30":
            if pr.get("step") not in (cfg.get("steps") or ()) or int(pr.get("phase") or 0) not in (1, 2):
                return None
            return {"pkg": pkg, "phase": int(pr["phase"]), "step": pr["step"]}, None
        if lever == "L31":
            out = {"kind": pr["kind"], "batch": int(pr["batch"]), "lab_inc": str(self.paths.lab_inc)}
            if pr["kind"] == "audit":
                srcs = sorted(pr.get("sources") or [])
                if not srcs:
                    return None
                out["sources"] = ",".join(srcs)
            return out, None
        if lever == "L32":
            arm = pr.get("arm")
            exp = (cfg.get("build_exps") or {}).get(arm)
            a = (st_.get("arms") or {}).get(arm) or {}
            if not exp or not a.get("images"):
                return None
            info.update(images=int(a["images"]), arm=arm)
            return {"pkg": pkg, "exp": exp, "arm": arm, "seeds": "0,1,2", "role": "baseline"}, exp
        if lever == "L32X":
            arm, verb = pr.get("arm"), pr.get("verb")
            if verb not in ("smoke", "train"):
                return None
            if verb == "smoke":
                return {"pkg": pkg, "verb": "smoke", "arm": arm}, None
            ls = ((st_.get("long") or {}).get("arms") or {}).get(arm) or {}
            done = {int(k) for k in ls.get("trained") or []}
            seeds = [x for x in (cfg.get("seeds") or [0, 1, 2]) if int(x) not in done]
            if not seeds:
                return None
            if ls.get("projection_h") is not None:
                info["projection_h"] = float(ls["projection_h"])
            out = {"pkg": pkg, "verb": "train", "arm": arm, "seeds": ",".join(str(int(x)) for x in seeds)}
            if int(ls.get("walltime_h") or 24) == 36:
                out["walltime_h"] = 36
            return out, None
        if lever == "L33":
            if pr.get("arm") not in (cfg.get("score_arms") or ()):
                return None
            return {"pkg": pkg, "arm": pr["arm"]}, None
        phase = int(pr.get("phase") or 0)
        if phase not in (1, 2):
            return None
        items = self._ds1()["items"]
        arms = set((cfg.get("verdict_arms") or {}).get(str(phase)) or ()) | set(
            (cfg.get("verdict_arms") or {}).get("2_substituted") or () if phase == 2 else ())
        mine = [k for k in items if k.split(":")[1:2] and k.split(":")[1] in arms and k.split(":")[0] in (
            "build", "score", "long")]
        refused = sorted(k for k in mine if items[k].get("state") == DS1_NOT_RUN_CAP)
        blocked = (self._ds1().get("runs_blocked") or {})
        failed = sorted(set(k for k in mine if items[k].get("state") == "failed") | {
            "build:%s" % a for a, e in (cfg.get("build_exps") or {}).items() if e in blocked and a in arms})
        return dict({"pkg": pkg, "phase": phase}, **({"cap_refused": ",".join(refused)} if refused else {}),
                    **({"ticker_failed": ",".join(failed)} if failed else {})), None

    def _ds1_cap_ok(self, lever, params, est, pr):
        """False (recorded 'not run (family cap)', one card, never retried) when est would take DS1 past its cap
        (260 GPU-h) or, a phase-1 item before verdict v2 or the date substitution, past the phase reserve's line
        (260 - 85.8 GPU-h); and for the items cut last when the cap binds (S1c-R, S2 at 1.2 M), past the cap less
        the long pair's remaining estimate. The family's GPU-h as this tick's context gives it, plus the estimates
        proposed this tick."""
        if lever == "L31":
            return True
        cfg = (self.dom or {}).get("ds1") or {}
        cap = float(cfg.get("cap_gpu_hours") or 260.0)
        line = cap - float(cfg.get("phase_reserve_gpu_hours") or 85.8)
        ctx = (self.ev.json(E.CONTEXT) or {}) if self.ev is not None else {}
        stg = ((ctx.get("stage") or {}).get("ds1") or {})
        spent = float(stg.get("spend_gpuh") or 0.0)
        tick = getattr(self, "_ds1_tick", None)
        if tick is None or tick[0] != self.utc:
            tick = self._ds1_tick = [self.utc, 0.0]
        after = spent + tick[1] + float(est or 0.0)
        phase = int(pr.get("phase") or 1)
        v2 = (((stg.get("status") or {}).get("verdicts") or {}).get("2") or {}).get("status") == "decided"
        sub = str(stg.get("today") or "") > str((cfg.get("dates") or {}).get("s2_freeze_by") or "9999")
        key = pr.get("item") or self._ds1_key(lever, params)
        why = None
        if after > cap + V3FAM_CAP_EPS:
            why = "would take DS1 from %.2f past its cap of %.0f GPU-h" % (spent + tick[1], cap)
        elif phase == 1 and not v2 and not sub and after > line + V3FAM_CAP_EPS:
            why = "would take DS1 from %.2f past %.1f GPU-h, the phase reserve's line (%.0f - %.1f kept for phase 2 " \
                  "and the reads)" % (spent + tick[1], line, cap, cap - line)
        elif key in ("build:S2", "build:S1CR"):
            costs = cfg.get("costs") or {}
            pair_left = 0.0 if (stg.get("items") or {}).get("long:S2X:train") == "done" else \
                3 * float(costs.get("long_train_hours") or 14.2)
            if after + pair_left > cap + V3FAM_CAP_EPS:
                why = "is cut first when the cap binds (DS.11): with the long pair's remaining %.1f GPU-h it would take " \
                      "DS1 from %.2f past %.0f" % (pair_left, spent + tick[1], cap)
        if why is None:
            tick[1] += float(est or 0.0)
            return True
        rec = self._ds1()["items"].setdefault(key, {})
        if rec.get("state") != DS1_NOT_RUN_CAP:
            self._v3fam_fold(rec)
            rec.update(state=DS1_NOT_RUN_CAP, refused_utc=self.utc, spent_gpuh=round(spent + tick[1], 4),
                       estimate=float(est or 0.0), spend=None)
            self._ledger("not_taken", lever=lever, ds1_item=key, reasons=["%s: %s (%.2f GPU-h) %s" % (
                DS1_NOT_RUN_CAP, key, float(est or 0.0), why)])
            self._card("escalation", "DS1's %s: %s" % (key, DS1_NOT_RUN_CAP),
                       "%s (%.2f GPU-h est.) %s (DS.0 item 10, DS.11). It is recorded '%s', keeps its place in DS1's "
                       "tables and is never retried." % (key, float(est or 0.0), why, DS1_NOT_RUN_CAP), lever=lever)
        return False

    def _ds1_started(self, it):
        p = it["proposal"]
        key, rec = self._ds1_item(it)
        if rec.get("proposal") == p["id"] and rec.get("spend") in ("open", "settled"):
            rec.update(state="running", lever=it["lever"], exp=p.get("child_exp"))
            return
        self._v3fam_fold(rec)
        cfg = (self.dom or {}).get("ds1") or {}
        rec.update(state="running", estimate=float(p.get("est_gpu_hours") or 0.0), spend="open", settled=None,
                   proposal=p["id"], lever=it["lever"], started_utc=self.utc, exp=p.get("child_exp"),
                   phase=int(p.get("ds1_phase") or 1),
                   build_job_gpuh=float((cfg.get("costs") or {}).get("build_job_hours") or 0.0)
                   if it["lever"] == "L32" else None)

    def _ds1_settle(self, it, rows, build_ended=False):
        """A DS1 job's spend settled from sacct once every job of it ended (V100 GPU-h); a build (L32) by its
        experiment's report and its build job's sacct, unless the job ended without the experiment."""
        if not rows or not all(r and r.get("elapsed_s") is not None for r in rows):
            return
        _key, rec = self._ds1_item(it)
        got = round(sum(float(r["elapsed_s"]) * float(r.get("gpu_count") or 1) / 3600.0 for r in rows), 4)
        if it["lever"] == "L32" and not build_ended:
            rec["build_job_gpuh"] = got               # the runs come from the experiment's report (_ds1_spend)
            return
        rec["settled"] = got
        rec["spend"] = "settled"

    def _ds1_done(self, it):
        """A DS1 item done: a data step goes back to idle (its next proposal takes a new id, so it can run again when
        its conditions call for it), every other item is done; phase 1's bytes counted at the item's estimate."""
        st = self.st
        p = it["proposal"]
        key, rec = self._ds1_item(it)
        cfg = (self.dom or {}).get("ds1") or {}
        f = self._ds1()
        sizes = cfg.get("storage_gb") or {}
        kind = {"L30": "data:%s" % (p.get("params") or {}).get("step"), "L32": "build",
                "L32X": "long:%s" % (p.get("params") or {}).get("verb"), "L33": "score", "L34": "verdict"}.get(
            it["lever"])
        charged = list(f.get("phase1_charged") or [])
        if kind and int(p.get("ds1_phase") or 1) == 1 and key not in charged:
            # each item's nominal bytes once (DS.11's bounds are per item: a data step that runs again, the pools
            # for sources decided later, writes within the same bound), so lane DATA's floor applies only past 60 GB
            # of what phase 1 can write, never by repetition
            f["phase1_gb"] = round(float(f.get("phase1_gb") or 0.0) + float(sizes.get(kind) or 0.0), 3)
            f["phase1_charged"] = sorted(charged + [key])
        rec["fails"] = 0
        if it["lever"] == "L30":
            rec.update(state="missing", last_done_utc=self.utc, runs=int(rec.get("runs") or 0) + 1)
        else:
            rec.update(state="done", done_utc=self.utc)
        if it["lever"] == "L31" and (p.get("params") or {}).get("kind") == "audit":
            # the set this render drew: its own proposal's sources (DR0 named them from the cluster's audit set),
            # never this tick's context, which need not hold DS1's status when a job's end is read (Revision 2,
            # item 3: on 2026-10-29 it held none, so [] was recorded and the same set was rendered on every tick)
            f["audit_render_for"] = ds1_audit_sources((p.get("params") or {}).get("sources"))
        # the next proposal of the same step takes a new id: the executor runs an id once
        akey = step_key(it["lever"], LS.policy_params(it["lever"], p.get("params") or {}))
        att = st.setdefault("attempts", {})
        att[akey] = int(att.get(akey) or 0) + 1
        return key

    def _ds1_failed(self, ln, it, why, charged, ended=None):
        """A DS1 item that ended without its effect: one card; proposed once more under a new id; after its second
        failure (ds1.max_failures) recorded failed, its id declined, DS1's other items go on. No lane failure, no
        stop-loss. A long-path seed killed at its walltime (sacct TIMEOUT) fails the arm at once."""
        st = self.st
        p = it["proposal"]
        lever, params = it["lever"], p.get("params") or {}
        cfg = (self.dom or {}).get("ds1") or {}
        key, rec = self._ds1_item(it)
        rec["fails"] = int(rec.get("fails") or 0) + 1
        if rec.get("spend") == "open":
            if ended is not None and lever == "L32":
                rec["settled"] = float(rec.get("build_job_gpuh") or ((cfg.get("costs") or {}).get("build_job_hours"))
                                       or 0.0)
            else:
                rec["settled"] = float(rec.get("estimate") or 0.0) if charged else 0.0
            rec["spend"] = "settled"
        walltime = lever == "L32X" and params.get("verb") == "train" and any(
            str(x).startswith("TIMEOUT") for x in it.get("ds1_states") or [])
        last = walltime or rec["fails"] >= int(cfg.get("max_failures") or 2)
        rec["state"] = "failed" if last else "failed_once"
        if walltime:
            rec["why"] = "walltime: a seed was killed at its limit before its export (DS.6)"
        if last:
            st["declined"] = (list(st.get("declined") or []) + [p["id"]])[-200:]
        else:
            st["failed_ids"] = (self._failed_ids() + [p["id"]])[-FAILED_IDS_KEEP:]
        self._ledger("failed", lane=ln, lever=lever, reasons=[_short(why, 1000)], proposal_id=p["id"],
                     charged=charged, record_only=True, ds1_item=key, fails=rec["fails"], walltime=walltime)
        title = RECORD_ONLY_TITLES[lever] % key.split(":", 1)[1]
        body = "%s. Record only: %s; DS1's other items go on (DS.9.3). A person may rerun it from the login node: %s" % (
            _short(why, 400), "after its second failure it is recorded failed and keeps its place in DS1's tables as "
            "failed" if last else "it is proposed once more under a new id", self._ds1_rerun(lever, params))
        self._card("escalation", "%s%s" % (title, ": recorded failed" if last else ""), body, lever=lever,
                   trigger=list(p.get("trigger") or []))
        self._clear(ln)

    @staticmethod
    def _ds1_rerun(lever, p):
        mod = "python -m weed_optimizer_framework.tools.inc2"
        if lever == "L30":
            return "sbatch -p GPU-shared run_inc2_build.sh inc2.ds1_data build --phase %s --step %s" % (
                p.get("phase"), p.get("step"))
        if lever == "L31":
            return "%s.ds1_sheets render --kind %s --batch %s (on the lab)" % (mod, p.get("kind"), p.get("batch"))
        if lever == "L32":
            return "sbatch -p GPU-shared run_inc2_build.sh inc2.baseline build --exp %s --ds1 %s --seeds 0,1,2 " \
                   "--role baseline" % (p.get("exp"), p.get("arm"))
        if lever == "L32X":
            return "sbatch -p GPU-shared run_inc2_long.sh inc2.ds1_long %s --arm %s%s" % (
                p.get("verb"), p.get("arm"), " --seed <k>" if p.get("verb") == "train" else "")
        if lever == "L33":
            return "sbatch -p GPU-shared run_inc2_build.sh inc2.ds1 score --arm %s" % p.get("arm")
        return "sbatch -p GPU-shared run_inc2_build.sh inc2.ds1 verdict --phase %s" % p.get("phase")

    def _ds1_follow(self, it):
        """(job name(s), record artifact) a DS1 item is followed by after an unknown outcome."""
        from . import stream_remote as SR
        prm = it["proposal"].get("params") or {}
        lv = it["lever"]
        if lv == "L30":
            return SR.DS1_DATA_JOB_NAME % (int(prm.get("phase") or 1), prm.get("step")), "splits/ds1/summary.json"
        if lv == "L33":
            return SR.DS1_SCORE_JOB_NAME % str(prm.get("arm")).lower(), "capacity/ds1_score_%s.json" % str(
                prm.get("arm")).lower()
        if lv == "L34":
            return SR.DS1_VERDICT_JOB_NAME % int(prm.get("phase") or 1), "capacity/ds1_v%d.json" % int(
                prm.get("phase") or 1)
        if lv == "L32X":
            arm = str(prm.get("arm")).lower()
            names = (SR.DS1_LONG_JOB_NAME % (arm, "smoke"),) if prm.get("verb") == "smoke" else tuple(
                SR.DS1_LONG_JOB_NAME % (arm, "s%d" % k) for k in (0, 1, 2))
            return names, SR.DS1_STATUS
        return "inc_build_%s" % prm.get("exp"), SR.DS1_STATUS

    def _ds1_record_done(self, it, rec):
        """A DS1 item followed after an unknown outcome is done once its record says so."""
        prm = it["proposal"].get("params") or {}
        lv = it["lever"]
        if lv == "L34":
            return rec.get("status") == "decided"
        if lv == "L33":
            return rec.get("status") == "complete"
        if lv == "L30":
            ls = rec.get("last_step") or {}
            return ls.get("phase") == prm.get("phase") and ls.get("step") == prm.get("step") and \
                str(ls.get("utc") or "") >= str(self._ds1_item(it)[1].get("started_utc") or "")
        if lv == "L32X":
            arm = ((rec.get("long") or {}).get("arms") or {}).get(prm.get("arm")) or {}
            if prm.get("verb") == "smoke":
                return bool(arm.get("smoke"))
            return sorted(int(k) for k in arm.get("trained") or []) == [0, 1, 2]
        return False

    def _ds1_overdue(self, it):
        """One card when an L32X item's jobs are still queued DS1_LONG_OVERDUE_H after its submission."""
        t0 = _secs(it.get("since_utc"))
        if it.get("ds1_overdue") or t0 is None or self.now - t0 < DS1_LONG_OVERDUE_H * 3600.0:
            return
        it["ds1_overdue"] = self.utc
        self._card("escalation", "DS1's long-path %s jobs are queued or running after %.0f h (L32X)" % (
            (it["proposal"].get("params") or {}).get("verb"), DS1_LONG_OVERDUE_H),
            "job(s) %s: each has a 24 h walltime (36 h for S2x's one re-submission), so a job still queued or running "
            "after %.0f h waits in the queue or outlived its limit; a person checks squeue and sacct. The TRAIN lane "
            "stays held until they end." % (",".join(str(j) for j in it.get("job_ids") or []), DS1_LONG_OVERDUE_H),
            lever="L32X")

    def _ds1_runs_blocked(self, fired):
        """DS1's build experiments whose runs are blocked (D5, not transient) or stale (D6 on STALE_TO_PAUSE
        snapshots) this tick: recorded with one card each (the arm counts as failed in DS1's plan and the verdict's
        argv names it: a failed arm never holds the other arms, DS.9.3); an experiment no longer blocked is cleared."""
        cfg = (self.dom or {}).get("ds1") or {}
        if not cfg:
            return
        exps = {v: k for k, v in (cfg.get("build_exps") or {}).items()}
        f = self._ds1()
        blocked = f.setdefault("runs_blocked", {})
        now = {d.get("exp"): d for d in fired if d.get("exp") in exps}
        for exp, d in sorted(now.items()):
            if exp in blocked:
                continue
            blocked[exp] = {"utc": self.utc, "id": d.get("id"), "summary": _short(d.get("summary"), 300)}
            self._card("escalation", "DS1's %s: its runs are %s (%s)" % (
                exps[exp], "blocked" if d.get("id") == "D5" else "stale", d.get("id")),
                "%s. DS1 counts the arm failed (DS.9.3: a failed arm never holds the other arms) and its phase verdict "
                "names it --ticker-failed build:%s, unless a person unblocks it first (inc.driver unblock --exp %s): "
                "once its runs advance it is running again." % (_short(d.get("summary"), 400), exps[exp], exp),
                lever="L32")
        for exp in list(blocked):
            if exp not in now:
                blocked.pop(exp)

    def _ds1_sheets_hook(self, params):
        """L31's lab hook: inc2.ds1_sheets render of one batch on the lab (it pulls its bundles from the cluster and
        pushes its render record back), a detached lab process."""
        job = "ds1sheets_%s_%s_%s" % (params.get("kind"), params.get("batch"), _sha([params, self.utc])[:10])
        argv = [sys.executable, "-m", "weed_optimizer_framework.tools.inc2.ds1_sheets", "render", "--kind",
                str(params.get("kind")), "--batch", str(params.get("batch")), "--lab-inc",
                str(params.get("lab_inc") or self.paths.lab_inc)] + (
            ["--sources", str(params["sources"])] if params.get("sources") else [])
        try:
            got = self.runner.launch(job, argv, timeout=LAB_TIMEOUT_S)
        except Exception as e:
            return {"ok": False, "error": "%s: %s" % (type(e).__name__, _short(e, 300))}
        return {"ok": True, "detached": True, "lab_job": got.get("job", job), "argv": argv}

    def _ds1_card_once(self, name, content, kind, title, detail, lever):
        f = self._ds1()
        mark = _sha(content)
        if f["cards"].get(name) == mark:
            return
        f["cards"][name] = mark
        self._card(kind, title, detail, lever=lever)

    def _ds1_cards(self, by):
        """DS1's cards (each raised once per content): the gate's missing pieces; what waits for a person (the
        time stamp, RA, the sheets to decide, the owner's audit, the MT-W check, LN, the re-audit of s001 and s002),
        each naming the exact step; the storage waits DR0 names; each decided verdict with the sealed reads' exact
        commands in DS.7.5's order."""
        cfg = (self.dom or {}).get("ds1") or {}
        if not cfg:
            return
        stg = ((self.ev.json(E.CONTEXT) or {}).get("stage") or {}) if self.ev is not None else {}
        fs = stg.get("ds1") or {}
        if not fs.get("amendment"):
            return
        mod = "python -m weed_optimizer_framework.tools.inc2"
        st = fs.get("status") or {}
        if not fs.get("never_cut"):
            self._ds1_card_once("never_cut", "deploy", "research", "DS1 waits for the cutter's never-cut change",
                                "The deployed inc2/step1_stream.py does not read DS1's never-train list (DS.9.1): no "
                                "DS1 item is proposed until a deploy ships it (deploy/deploy_funnel.sh).", "L30")
            return
        if not st.get("timestamp"):
            rv = list(cfg.get("revisions") or [])
            tm = st.get("timestamp_missing") if isinstance(st.get("timestamp_missing"), dict) else {}
            upgrade = ("Nothing of DS1 is proposed before it. A proof still pending in Bitcoin is upgraded later (ots "
                       "upgrade; ots verify's output recorded) and committed: the record names what each proof "
                       "stamps, which the upgrade does not change.")
            # one card per set of missing records (Revision 2, item 6): ds1/v1/timestamp.json is written once, so a
            # Revision it does not name is recorded by its own step, named by the Revision's own number, and only the
            # Revisions the cluster's status lists as unnamed are named (never one already recorded, which is refused)
            need = [r for r in rv if tm.get("record") and r.get("n") is not None and
                    int(r["n"]) in [int(x) for x in tm.get("revisions") or []]]
            if need:
                self._ds1_card_once("timestamp", "revisions:%s" % ",".join("%s=%s" % (r["n"], str(r.get("sha256"))[
                    :12]) for r in need), "research", "DS1 waits for its time-stamp record",
                    "DS.12 step 0a: ds1/v1/timestamp.json exists (it is written once) and names no record of %s. On "
                    "the login node, from the deployed tree's root: %s (ds1/v1/timestamp_rev<n>.json). %s" % (
                        ", ".join("Revision %s" % r["n"] for r in need), "; ".join(
                            "%s.ds1 timestamp-revision --revision %s --proof %s.ots --stamp-proof %s.ots" % (
                                mod, r["n"], r.get("file"), r.get("stamp_file")) for r in need), upgrade), "L30")
                return
            revs = " ".join("--revision-proof %s.ots --revision-stamp-proof %s.ots" % (r.get("file"), r.get(
                "stamp_file")) for r in rv)
            self._ds1_card_once("timestamp", "missing:%s" % ",".join(str(r.get("sha256"))[:12] for r in rv),
                                "research", "DS1 waits for its time-stamp record",
                                "DS.12 step 0a: on the login node, %s.ds1 timestamp --proof docs/prereg/"
                                "DS1_2026-10-29.md.ots --stamp-proof docs/prereg/DS1_2026-10-29.stamp.txt.ots %s "
                                "(ds1/v1/timestamp.json names the prereg's and each Revision's sha256 and proofs). %s"
                                % (mod, revs, upgrade), "L30")
            return
        steps = []
        if not st.get("ra"):
            steps.append("RA (DS.8.1, the delegate, no GPU): %s.ds1 ra on the login node (every segment report "
                         "with a gate record); every training item waits for it" % mod)
        sh = st.get("sheets") or {}
        for ph in ("1", "2"):
            srcs = ((st.get("phases") or {}).get(ph) or {}).get("sources") or {}
            # a source role C or quarantined whole needs no decision (it enters no DS1 manifest)
            und = sorted(s_ for s_, x in srcs.items() if not x.get("decided") and not x.get("role_c")
                         and not x.get("quarantined_whole"))
            if und:
                steps.append("sheets waiting (phase %s, the delegate): %s; the HTML batches are on the lab under %s, "
                             "to copy to %s; each saved decisions file is stored with %s.ds1_sheets record --file <it> "
                             "--decided-by delegate on the login node" % (ph, ", ".join(und[:12]),
                                                                         (cfg.get("lab") or {}).get("sheets_html"),
                                                                         (cfg.get("lab") or {}).get("owner_folder"),
                                                                         mod))
        for ph in ("1", "2"):
            srcs = ((st.get("phases") or {}).get(ph) or {}).get("sources") or {}
            # Revision 2, item 4: 'accept' on a source the model audit finds partially labelled waits for a new decision
            again = sorted(s_ for s_, x in srcs.items() if x.get("decided") and DS.DS1_DECIDE_AGAIN in str(x.get("why")
                                                                                                       or ""))
            if again:
                steps.append("a new sheet decision waits (phase %s; DS.5.1 item 4: the model audit finds them "
                             "partially labelled, so they enter only as 'accept for training only'): %s; stored with "
                             "%s.ds1_sheets record --file <it> --decided-by delegate, or --decided-by human:<his id> "
                             "for a source whose owner's audit decision governs it (the delegate's does not override "
                             "his)" % (ph, ", ".join(again[:12]), mod))
        if sh.get("audit_pending"):
            steps.append("owner audit waiting (DS.0 item 4): %s; the audit sheet batch is rendered on the lab; his "
                         "decisions are stored with %s.ds1_sheets record --file <it> --decided-by human:<his id>"
                         % (", ".join((sh.get("audit_pending") or [])[:12]), mod))
        waiting_mtw = sorted(s_ for ph in ("1", "2") for s_, x in (((st.get("phases") or {}).get(ph) or {}).get(
            "sources") or {}).items() if x.get("mtw_draft") and not x.get("checked"))
        if sh.get("mtw_batches") and (waiting_mtw or not st.get("mtw_v1_1")):
            steps.append("MT-W check waiting (DS.10 item 4, the owner or a person he names): the mtw sheets on the lab"
                         "%s; recorded with %s.ds1_sheets record" % ((" (MT-W v2: %s)" % ", ".join(waiting_mtw[:8]))
                                                                    if waiting_mtw else "", mod))
        if sh.get("ln_batches") and not sh.get("ln_recorded"):
            steps.append("the LN sheet (DS.8.2, the owner, 55-75 min): ln batch 1 on the lab; then %s.ds1 ln" % mod)
        if steps:
            self._ds1_card_once("steps", steps, "research", "DS1 waits for a person's step%s" % (
                "s" if len(steps) > 1 else ""), "Every other DS1 item, and every stream and DATA item, goes on (DS.9.3: "
                "a card holds no lane). Waiting: " + "; ".join(steps) + ".", "L31")
        segs = list(cfg.get("reaudit_segments") or [])
        if segs:
            self._ds1_card_once("reaudit", segs, "research", "DS1: re-audit %s with AS (DS.10 item 2)" % ", ".join(segs),
                                "Move each %s audit/label_audit.json aside (never delete) and re-submit L4's job "
                                "script; remote.py refuses a rerun over an existing record (AS)." % ", ".join(segs),
                                "L4")
        # the storage waits DR0 names: one card when an item starts waiting (not again while it keeps waiting, so a
        # quota that moves each snapshot raises no new card); an item that stops waiting may raise one later
        sw = {k: why for k, why in (((by.get("DR0") or {}).get("detail") or {}).get("ds1_waits") or {}).items()
              if str(why).startswith("storage")}
        f = self._ds1()
        prev = set(f.get("storage_waits") or [])
        new = sorted(set(sw) - prev)
        if new:
            self._card("quota", "DS1's %s wait%s for storage" % (", ".join(new), "s" if len(new) == 1 else ""),
                       "%s (DS.0 item 21). A person frees space or the item%s wait%s; every other item goes on." % (
                           "; ".join("%s: %s" % (k, sw[k][len("storage: "):] if sw[k].startswith("storage: ") else sw[k])
                                     for k in new), "s" if len(new) > 1 else "", "" if len(new) > 1 else "s"),
                       lever="L30")
        f["storage_waits"] = sorted(sw)
        for ph in ("1", "2"):
            v = self._artifact(((cfg.get("verdict_records") or {}).get(ph)) or "capacity/ds1_v%s.json" % ph) or {}
            if v.get("status") != "decided":
                continue
            lines = []
            for pid, r in sorted((v.get("primaries") or {}).items()):
                if r.get("status") == "decided":
                    lines.append("%s %s over %s: D %s, z %s (bar %s), 90%% interval %s -> %s" % (
                        pid, r.get("x"), r.get("y"), _fmt4(r.get("diff")), _fmt4(r.get("z")), _fmt4(r.get("threshold")),
                        [_fmt4(x) for x in r.get("interval_90") or []], r.get("reading")))
                else:
                    lines.append("%s: not run (%s)" % (pid, _short(r.get("why"), 120)))
            if v.get("n1"):
                lines.append("N1 non-inferior: %s (lower 90%% end %s); H1 fires: %s" % (
                    (v.get("n1") or {}).get("non_inferior"), _fmt4((v.get("n1") or {}).get("lower_90")),
                    (v.get("h1") or {}).get("fires")))
            for o in (v.get("reads") or {}).get("order") or []:
                lines.append("Read %s on %s (%s): %s.ds1 test-read --arm %s --exam %s --read-by owner|delegate; the "
                             "person named submits the printed sbatch argv with INCAP_DECIDED_BY naming them; then "
                             "%s.ds1 test-report --arm %s --exam %s" % (o.get("arm"), o.get("exam"), o.get("kind"),
                                                                       mod, o.get("arm"), o.get("exam"), mod,
                                                                       o.get("arm"), o.get("exam")))
            self._ds1_card_once("verdict:%s" % ph, lines, "research", "DS1's verdict v%s is decided" % ph,
                                "; ".join(lines) + ". Nothing switches (DS.0 item 13).", "L34")

    def _stream_quarantined(self):
        """{source: {cite, seq, utc}} the stream's queue summary (inc2.stream's ledger) records quarantined."""
        q = self._artifact("stream/%s/queue_summary.json" % self.sid) if self.st is not None else None
        rec = q.get("quarantined_sources") if isinstance(q, dict) else None
        if isinstance(rec, list):
            return {str(x): {} for x in rec}
        return {str(k): (v if isinstance(v, dict) else {}) for k, v in rec.items()} if isinstance(rec, dict) else {}

    def _resync_quarantines(self):
        """Record as quarantined every source the stream's ledger quarantines while this state says otherwise (an
        L17 admit done after the same tick's L24 had overwritten it: three D28 leaks on 2026-10-06/07, one on
        2026-10-01): status, cite, the stream ledger's seq and utc; its 'quarantine pending' reason dropped. One
        ledger event per source. A closed source stays closed. Returns the sources resynced."""
        out = []
        for src, rec in sorted(self._stream_quarantined().items()):
            s = (self.st.get("sources") or {}).get(src)
            if not isinstance(s, dict) or s.get("status") in ("quarantined", "closed"):
                continue
            prev = s.get("status")
            s.update(status="quarantined", cite=rec.get("cite") or s.get("cite"), quarantined_seq=rec.get("seq"),
                     quarantined_stream_utc=rec.get("utc"), quarantine_resynced_utc=self.utc)
            if not s.get("quarantined_utc"):
                s["quarantined_utc"] = rec.get("utc") or self.utc
            s.pop("held_reason", None)
            self._ledger("source_quarantine_resynced", source=src, cite=rec.get("cite"), seq=rec.get("seq"),
                         previous=prev, reasons=["the stream ledger quarantines %s (seq %s, %s, %s); this state said %r"
                                                 % (src, rec.get("seq"), rec.get("cite"), rec.get("utc"), prev)])
            out.append(src)
        self._resync_lifts()
        return out

    def _resync_lifts(self):
        """A quarantine a person lifted (inc2.stream unquarantine, by human:): once the stream's queue summary,
        read whole, no longer lists the source, this state's 'quarantined' record of an admitted source, taken
        before that lift, becomes 'admitted' again (one event), so that blames committed after the lift can quarantine it again (D31 counts
        only those; L24 under a new id). A summary that cannot be read, or a source with nothing admitted, is left
        as it is."""
        q = self._artifact("stream/%s/queue_summary.json" % self.sid)
        if not isinstance(q, dict) or not isinstance(q.get("quarantined_sources"), (dict, list)):
            return
        qs = self._stream_quarantined()
        lifts = {}
        for e in self._artifact("stream/%s/ledger.jsonl" % self.sid) or []:
            if isinstance(e, dict) and e.get("event") == "unquarantine" and e.get("source") \
                    and str(e.get("by") or "").startswith("human:"):
                lifts[str(e["source"])] = e
        for src, e in sorted(lifts.items()):
            s = (self.st.get("sources") or {}).get(src)
            if src in qs or not isinstance(s, dict) or s.get("status") != "quarantined" \
                    or not (s.get("admitted_batches") or s.get("admit_done_utc")):
                continue
            if not e.get("utc") or str(e["utc"]) <= str(s.get("quarantined_utc") or ""):
                continue                          # a quarantine this state took after that lift (an L24 that ran
                #                                   since the snapshot): the next snapshot lists it
            s.update(status="admitted", lifted_by=e.get("by"), lifted_utc=e.get("utc"), lifted_seq=e.get("seq"))
            self._ledger("source_lift_resynced", source=src, decided_by=e.get("by"), seq=e.get("seq"),
                         reasons=["a person lifted the quarantine of %s (stream ledger seq %s, %s): admitted again; "
                                  "blames committed after the lift may quarantine it again" % (src, e.get("seq"),
                                                                                              e.get("utc"))])

    def _v3fam_prov_refusal(self, payload, it, ids):
        """The refusal line of a family job's provenance (the snapshot's builds, remote.status): its last attempt's,
        when that attempt is one of this item's jobs (ids), or, after a submission whose outcome was unknown (ids
        None), when it started after this item was submitted; else ''."""
        builds = ((payload or {}).get("status") or {}).get("builds") or {}
        lever, prm = it["lever"], it["proposal"].get("params") or {}
        if lever == "L23J":
            name = "v3fam_score_%s" % str(prm.get("arm") or "").lower()
        elif lever == "L23I":
            name = prm.get("exp") or it["proposal"].get("child_exp")
        else:
            name = V3FAM_PROV.get(lever)
        rec = builds.get(name) if name else None
        if not isinstance(rec, dict) or not rec.get("refusal"):
            return ""
        if ids is not None:
            if str(rec.get("job_id") or "").split("_")[0] not in {str(j).split("_")[0] for j in ids}:
                return ""
        else:
            t0 = it.get("since_utc") or it.get("started_utc")
            if not t0 or str(rec.get("started_utc") or "") < str(t0):
                return ""
        return _refusal_text(rec.get("refusal"))

    def _v3fam_platform_refusal(self, ln, it, why, charged):
        """A family job refused by a precondition of the platform's own making (deviation 5,
        V3FAM_PLATFORM_REFUSAL_RE): a reproduction record taken by other code than the deployed one or not in place,
        an E3-frozen file changed, the E-series records changed under it, a superseded record that cannot be moved.
        It is no failure of the arm: its fails are not counted and it is never recorded failed, so it ends nothing
        and is never named in the verdict's --ticker-failed; its GPU-h stay counted against the family cap
        (prior_gpuh); its id is retired; it is held (DS.V3FAM_HELD_PLATFORM: neither due nor ended, so the verdict
        waits) and released once the snapshot reports the reproduction records current (_v3fam_release). One card
        naming the remedy; after V3FAM_PLATFORM_MAX such refusals of one item it stays held for a person (one
        escalation card), never looped. Deviation 6: a refusal of a deterministic check (V3FAM_PERSON_REFUSAL_RE:
        deviation 6's loader check, E3's ground-truth and frame checks, which read the exam and the loader, never the
        arm's weights) is held for a person at the first refusal (held_for_person; never released, never proposed
        again by itself): a recorded deviation's reopen block proposes it again. No lane failure, no stop-loss."""
        st = self.st
        p = it["proposal"]
        lever, params = it["lever"], p.get("params") or {}
        key = self._v3fam_key(lever, params)
        rec = self._v3fam_item(lever, params)
        if rec.get("spend") == "open":
            rec["settled"] = float(rec.get("estimate") or 0.0) if charged else 0.0
            rec["spend"] = "settled"
        self._v3fam_fold(rec)
        rec["platform_refusals"] = int(rec.get("platform_refusals") or 0) + 1
        person = bool(V3FAM_PERSON_REFUSAL_RE.search(str(why or "")))
        if person:
            rec["held_for_person"] = True
        final = person or rec["platform_refusals"] >= V3FAM_PLATFORM_MAX
        rec.update(state=DS.V3FAM_HELD_PLATFORM, proposal=None, held_utc=self.utc,
                   why="refused by a precondition of the platform's own making: %s" % _short(why, 300))
        st["failed_ids"] = (self._failed_ids() + [p["id"]])[-FAILED_IDS_KEEP:]
        self._ledger("v3fam_platform_refusal", lane=ln, lever=lever, proposal_id=p["id"], v3fam_item=key,
                     record_only=True, charged=charged, platform_refusals=rec["platform_refusals"], final=final,
                     held_for_person=person, reasons=[_short(why, 1000)])
        title = RECORD_ONLY_TITLES[lever]
        title = (title % (params.get("arm") or params.get("verb"))) if "%s" in title else title
        title = title.replace(" failed (", " refused by the platform's own precondition (")
        remedy = ("The family's reproduction records (v3fam/v1/equivalence) pin the sha256 of the code that took them "
                  "(inc2.v3fam.code_shas: the six E3-frozen files and inc2/v3fam.py; the domain's "
                  "v3fam.equivalence_code). The remedy is a recorded deviation: restore the code they pin byte for "
                  "byte (deviation 5 did, moving the verdict's CLI to inc2/v3fam_verdict.py), or move the records "
                  "aside byte for byte and retake them (inc2.v3fam equivalence --item ...) before any family score; "
                  "deploy_funnel.sh refuses a change to that code while records exist.")
        if V3FAM_LOADER_REFUSAL in str(why or ""):
            remedy = ("Deviation 6's loader check (inc2/v3fam_loader.py; its record v3fam/v1/exif/loader_shapes.json) "
                      "found an exam image whose label-cache size, loader frame or letterbox is not as that record "
                      "proves, a pass the fix did not see whole, or an identity record not taken under the fix. "
                      "It is the platform's loader or records, not the arm's detectors; a person finds why, and a "
                      "recorded deviation moves the records aside or changes the check.")
        elif person:
            remedy = ("E3's ground-truth and frame checks (inc2/twostage.py, unedited) found a ground-truth box that "
                      "does not come back through the geometry to its label, a box without its label row, or an image "
                      "whose EXIF tag or transposed size is not the frame the loader read. They read the exam's labels "
                      "and image headers and the loader's batch geometry, never the arm's weights: the platform's "
                      "loader or exam, not the arm (deviation 6, which found one such defect on 2026-10-25). A person "
                      "finds why; a recorded deviation fixes it and its reopen block proposes the item again.")
        if person:
            self._card("escalation", "%s: held for a person, not the arm's failure" % title,
                       "%s. A deterministic check of the platform's own making refused it (deviation 6): its fails "
                       "are not counted, it is never recorded failed or passed to the verdict as failed, it ends "
                       "nothing, and its GPU-h stay counted against the family cap. The same code on the same exam "
                       "refuses the same way, so it is held for a person at once and never proposed again by itself; "
                       "the family's verdict waits for it. %s" % (_short(why, 400), remedy), lever=lever,
                       trigger=list(p.get("trigger") or []))
        elif final:
            self._card("escalation", "%s: held after %d refusals" % (title, rec["platform_refusals"]),
                       "%s. Refused %d times by a precondition of the platform's own making although the snapshot "
                       "reported the reproduction records current: it stays held (no fails counted, nothing ended), "
                       "the family's verdict waits for it, and a person finds why. %s" % (
                           _short(why, 400), rec["platform_refusals"], remedy), lever=lever,
                       trigger=list(p.get("trigger") or []))
        else:
            self._card("escalation", "%s: held, not the arm's failure" % title,
                       "%s. This is no failure of the arm (deviation 5): its fails are not counted, it is never "
                       "recorded failed or passed to the verdict as failed, and it ends nothing; its GPU-h stay counted "
                       "against the family cap. It is held, and no family score is proposed while the snapshot reports "
                       "a record taken by other code; once the records are current it is proposed again under a new "
                       "id. %s" % (_short(why, 400), remedy), lever=lever, trigger=list(p.get("trigger") or []))
        self._clear(ln)

    def _v3fam_rerun(self, lever, params):
        """The command a person submits to rerun a family item by hand (an sbatch job, never a foreground run on the
        login node), with INCAP_DECIDED_BY naming them and sbatch's --export naming that variable (Bridges-2's
        SBATCH_EXPORT=NONE passes a job none of the submitter's variables sbatch does not name; deviation 3,
        2026-10-12); L23K's train one sbatch per seed not yet trained."""
        sb = "INCAP_DECIDED_BY=<the person> sbatch --export=INCAP_DECIDED_BY -p GPU-shared"
        if lever == "L23H":
            return "%s --job-name=inc_build_v3fam_data run_inc2_build.sh inc2.v3fam_data build" % sb
        if lever == "L23J":
            arm = str(params.get("arm"))
            return "%s --job-name=inc_build_v3fam_score_%s run_inc2_build.sh inc2.v3fam score --arm %s" % (
                sb, arm.lower(), arm)
        if lever == "L23L":
            cr = (" --cap-refused %s" % params["cap_refused"]) if params.get("cap_refused") else ""
            cr += (" --ticker-failed %s" % params["ticker_failed"]) if params.get("ticker_failed") else ""
            return "%s --job-name=inc_build_v3fam_v1 run_inc2_build.sh inc2.v3fam_verdict verdict%s" % (sb, cr)
        if lever == "L23I":
            return ("%s --job-name=inc_build_%s run_inc2_build.sh inc2.baseline build --exp %s --v3fam %s --seeds "
                    "0,1,2 --role baseline" % (sb, params.get("exp"), params.get("exp"), params.get("arm")))
        if params.get("verb") == "smoke":
            return "%s --job-name=inc_detr_v3fam_smoke run_inc2_detr.sh inc2.v3fam_detr smoke" % sb
        from . import stream_remote as SR
        fam = (self.dom or {}).get("v3fam") or {}
        st_ = self._artifact(fam.get("status") or SR.V3FAM_STATUS) or {}
        done = {str(int(k)) for k in ((st_.get("detr") or {}).get("trained") or [])}
        seeds = [x for x in str(params.get("seeds") or "0,1,2").split(",") if x != "" and x not in done]
        return "; ".join("%s --job-name=inc_detr_v3fam_s%s run_inc2_detr.sh inc2.v3fam_detr train --seed %s"
                         % (sb, k, k) for k in seeds) or "nothing: every seed is trained"

    def _v3fam_record_done(self, it, rec):
        """A family item followed after an unknown outcome is done once its record says so."""
        lever = it["lever"]
        if lever == "L23L":
            return rec.get("status") == "decided"
        if lever == "L23K":
            det = rec.get("detr") or {}
            if (it["proposal"].get("params") or {}).get("verb") == "smoke":
                return bool(det.get("smoke"))
            return sorted(int(k) for k in det.get("trained") or []) == [0, 1, 2]
        if lever == "L23J":
            return rec.get("status") in ("complete", "not run")
        return rec.get("status") == "complete"

    def _v3fam_overdue(self, it):
        """One card when an L23K item's jobs are still queued V3FAM_DETR_OVERDUE_H after its submission."""
        t0 = _secs(it.get("since_utc"))
        if it.get("v3fam_overdue") or t0 is None or self.now - t0 < V3FAM_DETR_OVERDUE_H * 3600.0:
            return
        it["v3fam_overdue"] = self.utc
        self._card("escalation", "V3F's RF-DETR %s jobs are queued or running after %.0f h (L23K)" % (
            (it["proposal"].get("params") or {}).get("verb"), V3FAM_DETR_OVERDUE_H),
            "job(s) %s: each has a 36 h walltime (run_inc2_detr.sh), so a job still queued or running after %.0f h "
            "waits in the queue or outlived its limit; a person checks squeue and sacct. The TRAIN lane stays held "
            "until they end." % (",".join(str(j) for j in it.get("job_ids") or []), V3FAM_DETR_OVERDUE_H),
            lever="L23K")

    @staticmethod
    def _v3fam_storage_need(fam, fs):
        """GB the family's items that are not yet terminal will write (stream_domains' v3fam.storage_gb per kind)."""
        sizes = fam.get("storage_gb") or {}
        need = 0.0
        if not DS.v3fam_terminal(fs.get("data")):
            need += float(sizes.get("data") or 0.0)
        need += sum(float(sizes.get("build") or 0.0) for v in (fs.get("build") or {}).values()
                    if not DS.v3fam_terminal(v))
        need += sum(float(sizes.get("score") or 0.0) for v in (fs.get("score") or {}).values()
                    if not DS.v3fam_terminal(v))
        for k in ("smoke", "train"):
            if not DS.v3fam_terminal((fs.get("detr") or {}).get(k)):
                need += float(sizes.get("detr_%s" % k) or 0.0)
        if not DS.v3fam_terminal(fs.get("verdict")):
            need += float(sizes.get("verdict") or 0.0)
        return need

    def _v3fam_storage_card(self, f, fam, fs):
        """One card when the family's remaining items need more than /ocean holds above D27's 3 % floor and the 30 GiB
        margin (V3F.6.4's per-item rule would stall the family midway): the shortfall a person frees first. Raised
        again only when the shortfall grows by 1 GB or more."""
        ctx = (self.ev.json(E.CONTEXT) or {}) if self.ev is not None else {}
        qu = ctx.get("quota") or {}
        quota, free = _num(qu.get("quota_gb")), _num(qu.get("free_gb"))
        if quota is None or free is None:
            return
        need = self._v3fam_storage_need(fam, fs)
        floor = float(fam.get("storage_floor_share") or 0.03) * quota
        margin = float(fam.get("storage_margin_gib") or 30.0) * 1.073741824
        room = free - floor - margin
        short = need - room
        if short <= 0:
            f.pop("storage_projection", None)
            return
        if f.get("storage_projection") is not None and short < float(f["storage_projection"]) + 1.0:
            return
        f["storage_projection"] = round(short, 1)
        self._card("quota", "V3F's remaining items need about %.1f GB more on /ocean than it holds above the floor "
                   "and margin" % short,
                   "/ocean: %.0f GB free - the 3 %% floor %.0f GB - the 30 GiB margin %.0f GB leaves %.1f GB; the "
                   "family's items not yet done will write about %.1f GB (stream_domains' v3fam.storage_gb), so an "
                   "item would wait for storage midway (V3F.6.4). A person frees at least %.1f GB on /ocean (the "
                   "Hugging Face cache counts there too) before then." % (free, floor, margin, room, need, short),
                   lever="L23H")

    def _v3fam_cards(self, by):
        """The family's cards (each raised once per content), once the deployed tree holds the amendment and its
        prerequisites hold (E3's verdict decided, the zoo complete): its person steps while any is pending; each
        storage wait DR0 names; D5b's block; the decided verdict with the exact commands of every sealed-test read
        in the order it fixed."""
        fam = (self.dom or {}).get("v3fam") or {}
        if not fam:
            return
        f = self._v3fam()
        stg = ((self.ev.json(E.CONTEXT) or {}).get("stage") or {}) if self.ev is not None else {}
        fs = stg.get("v3fam") or {}
        e3v = self._artifact(((fam.get("requires") or {}).get("e3_record")) or "capacity/e3_v1.json") or {}
        if stg.get("e3") != "done" or e3v.get("status") != "decided" or stg.get("zoo") != "done" \
                or not fs.get("amendment"):
            return
        mod = "python -m weed_optimizer_framework.tools.inc2"
        steps = []
        if not fs.get("timestamp"):
            steps.append("write v3fam/v1/timestamp.json (the OpenTimestamps proofs' file names and sha256s, V3F.12: "
                         "the amendment's and, under 'revisions', the stamp of %s)" % (fam.get("revision_file")
                                                                                       or "the Revisions"))
        if not fs.get("sweep"):
            steps.append("D0-S (the delegate): INCAP_DECIDED_BY=assistant:claude-delegate sbatch "
                         "--export=INCAP_DECIDED_BY -p GPU-shared "
                         "--job-name=inc_build_v3fam_sweep run_inc2_build.sh inc2.v3fam_sweep run; then, per verified "
                         "pair, %s.v3fam_sweep verdict --pair <id> --copy yes|no --read-by owner|delegate" % mod)
        if not fs.get("exif"):
            steps.append("the EXIF frame pre-flight before D1's scores: %s.v3fam exif-preflight, then %s.v3fam "
                         "exif-verdict --frame yes|no --read-by owner|delegate" % (mod, mod))
        if fs.get("score") and any(str(v).startswith("not run (download") for v in (fs.get("score") or {}).values()):
            steps.append("D2's weights: %s.v3fam_embed plan, the downloads on the login node, then %s.v3fam_embed "
                         "downloads --commands <log>" % (mod, mod))
        if steps and f.get("steps_card") != steps:
            f["steps_card"] = steps
            self._card("research", "V3F waits for a person's step%s" % ("s" if len(steps) > 1 else ""),
                       "The v3 experiment family (record only) proposes nothing until: " + "; ".join(steps)
                       + ". D7 (the lab RTX 3060's latency) is the delegate's at any time after the downloads.",
                       lever="L23H")
        self._v3fam_storage_card(f, fam, fs)
        eq = fs.get("equivalence") or {}
        mark = [eq.get("fingerprint"), eq.get("stale"), eq.get("missing")] if eq.get("state") == "stale" else None
        if mark and f.get("equiv_card") != mark:
            f["equiv_card"] = mark
            self._card("escalation", "V3F's family scores wait: reproduction records taken by other code or not in "
                       "place", "Taken by other code than the deployed one: %s; not in place: %s (the deployed code's "
                       "fingerprint %s). inc2.v3fam.require_equivalence refuses every family score resting on them "
                       "('the code changed since ... ran'), so the platform proposes no family score until a recorded "
                       "deviation restores the code they pin (inc2.v3fam.code_shas, the domain's v3fam.equivalence_code) "
                       "byte for byte, or moves them aside byte for byte and retakes them (deviation 5)." % (
                           ", ".join(eq.get("stale") or []) or "none", ", ".join(eq.get("missing") or []) or "none",
                           str(eq.get("fingerprint"))[:16]), lever="L23J")
        elif not mark:
            f.pop("equiv_card", None)
        # D6's smoke ended its seeds by the walltime rule: one card (deviation 5: on 2026-10-21 that outcome was
        # recorded with no card, and its epoch time was the whole fit, setup included; it stayed unseen for days)
        det = fs.get("detr") or {}
        if str(det.get("smoke") or "").startswith("not run") and f.get("detr_card") != det.get("smoke_sha256"):
            f["detr_card"] = det.get("smoke_sha256")
            self._card("research", "V3F: D6 %s by its smoke" % det.get("smoke"),
                       "D6's smoke (v3fam/v1/d6/smoke/smoke.json, sha256 %s) recorded %s: epoch time %s s x 50 + 0.5 h "
                       "projects %s h against the 28.8 h rule (V3F.5 D6), so its three seeds are not proposed; D6 stays "
                       "in C and in the dev Holm set, and P5 is reported 'not run' with this reason. The smoke records "
                       "the epoch's own time (from the first training epoch's start to the end of fit) and the "
                       "one-time setup before it apart (deviation 5)." % (
                           str(det.get("smoke_sha256"))[:12], det.get("smoke"), det.get("epoch_seconds"),
                           det.get("projection_h")), lever="L23K")
        for k, why in sorted((((by.get("DR0") or {}).get("detail") or {}).get("v3fam_waits") or {}).items()):
            if str(why).startswith("storage") and (f.setdefault("wait_cards", {})).get(k) != why:
                f["wait_cards"][k] = why
                self._card("quota", "V3F's %s waits for storage" % k, "%s (V3F.6.4: 30 GiB above the 3 %% floor for "
                           "the item's bytes); a person frees space" % why, lever="L23H")
        blocked = sorted((fam.get("blocked_builds") or {}).items())
        if blocked and not f.get("blocked_card"):
            f["blocked_card"] = self.utc
            self._card("research", "V3F: D5b's builds wait for a Revision", "%s: %s. P8 and P9 stay in the probe's "
                       "Holm set and are reported 'not run' until then." % (", ".join(k for k, _v in blocked),
                                                                           blocked[0][1]), lever="L23I")
        v = self._artifact(fam.get("verdict_record") or "capacity/v3fam_v1.json") or {}
        if v.get("status") != "decided" or f.get("verdict_card"):
            return
        lines = []
        for pid, r in sorted((v.get("primaries") or {}).items()):
            d = r.get("decision") or {}
            h = (v.get("holm") or {}).get(pid) or {}
            if r.get("status") == "decided":
                lines.append("%s %s over %s: D %s, z %s, 90%% interval [%s, %s] -> %s%s%s" % (
                    pid, r.get("x_label") or r.get("x"), r.get("y_label") or r.get("y"), _fmt4(d.get("diff")),
                    _fmt4(d.get("z")), _fmt4((d.get("interval_90") or [None, None])[0]),
                    _fmt4((d.get("interval_90") or [None, None])[1]), d.get("outcome"),
                    " (%s)" % ", ".join(d.get("flags") or []) if d.get("flags") else "",
                    "; family-confirmed" if h.get("family_confirmed") else ""))
            else:
                lines.append("%s: not run (%s)" % (pid, _short(r.get("why"), 120)))
        ch = v.get("challengers") or {}
        lines.append("Tier C: C = %s, bar z > %s; opening a read: %s" % (ch.get("C"), _fmt4(ch.get("bar")),
                                                                         ", ".join(ch.get("opening") or []) or "none"))
        order = (v.get("reads") or {}).get("order") or []
        for o in order:
            lines.append("Read %s (%s): %s.v3fam test-read --arm %s --read-by owner|delegate; the person named "
                         "submits the printed sbatch argv (it names INCAP_DECIDED_BY in --export) with "
                         "INCAP_DECIDED_BY naming them (the owner's account, or assistant:claude-delegate); then "
                         "%s.v3fam test-report --arm %s" % (o.get("arm"), o.get("kind"), mod, o.get("arm"), mod,
                                                           o.get("arm")))
        lines.append("After the reads, the re-analysis (no inference): INCAP_DECIDED_BY=<the person> sbatch "
                     "--export=INCAP_DECIDED_BY -p GPU-shared --job-name=inc_build_v3fam_retest run_inc2_build.sh "
                     "inc2.v3fam reanalyse-test --read-by owner|delegate. Nothing switches; the headline stays E3-B "
                     "(V3F.3.6).")
        f["verdict_card"] = self.utc
        cont = v.get("continues") or {}
        second = bool(cont.get("continuation"))
        if second:
            lines.insert(0, "The family's second continuation verdict over every arm (deviation 5); the verdict of "
                            "2026-10-13 (%s) and its continuation of 2026-10-22 (%s) stay recorded as they are, and "
                            "every comparison either decided is reproduced here." % (
                                cont.get("verdict") or "capacity/v3fam_v1.json",
                                cont.get("continuation") or "capacity/v3fam_v1_cont.json"))
        elif cont:
            lines.insert(0, "The family's continuation verdict over every arm (deviation 4); the verdict of 2026-10-13 "
                            "(%s) stays recorded as it is, and every comparison it decided is reproduced here."
                         % (cont.get("verdict") or "capacity/v3fam_v1.json"))
        self._card("research", "V3F's %sverdict is decided (%s)" % (
            "second continuation " if second else "continuation " if cont else "", v.get("decided_on", "dev")),
            "\n".join(lines), lever="L23L", trigger=["L23L"])

    def _measure_exps(self):
        """The measurement arms' experiments (baselines marked measure), V3F's build experiments (2026-10-11) and
        DS1's (2026-10-29): record only, so their D5 and D6 are a person's card, never a pause."""
        out = {b["exp"] for b in ((self.dom or {}).get("baselines") or {}).get("items") or [] if b.get("measure")}
        out |= set((((self.dom or {}).get("v3fam") or {}).get("build_exps") or {}).values())
        return out | set((((self.dom or {}).get("ds1") or {}).get("build_exps") or {}).values())

    def _e1_items(self):
        """{'A': item, 'B': item} of E1's baselines (stream-domain e1.arms), or {}."""
        e1 = (self.dom or {}).get("e1") or {}
        items = {b["id"]: b for b in ((self.dom or {}).get("baselines") or {}).get("items") or []}
        return {k: items[i] for k, i in sorted((e1.get("arms") or {}).items()) if i in items}

    def _e1_exp(self, arm):
        """The experiment of E1's arm 'A' or 'B', or None."""
        return (self._e1_items().get(arm) or {}).get("exp")

    def _native_item(self, params):
        """The measurement baseline an L23N item rescores (by its --exp), or None."""
        return next((b for b in ((self.dom or {}).get("baselines") or {}).get("items") or []
                     if b.get("measure") and b.get("exp") == (params or {}).get("exp")), None)

    def _measure_card(self, d):
        """A measurement arm's D5 (a block that is not transient) or D6 (a
        stale advance): a person's card, once per block and generation, and
        the stream runs on; the experiment stays built, so DR0 does not
        propose it again."""
        exp = d.get("exp")
        gen = (((self.st.get("history") or {}).get(exp) or [{}])[-1]).get("generation")
        if self._once("measure_health:%s:%s" % (d["id"], exp), [(d.get("detail") or {}).get("units"), gen],
                      "measure_health", exp=exp, trigger=[d["id"]], summary=_short(d.get("summary"), 300)):
            self._card("escalation", "Measurement arm %s: %s" % (exp, d["id"]),
                       "%s. The arm is record only and the stream runs on; a person unblocks it (inc.driver "
                       "unblock) or leaves it" % _short(d.get("summary"), 400), trigger=[d["id"]])


# ------------------------------------------------------------------ status
def summary(cfg, st):
    """A compact, page-ready view of one stream campaign."""
    st = st or {}
    lanes = st.get("lanes") or {}
    alarm = "crit" if (cfg.get("paused_reason") or (st.get("paused") or {}).get("reason")) else \
        "off" if not cfg.get("enabled") else "warn" if (st.get("errors") or any(
            (lanes.get(ln) or {}).get("hold") for ln in LANES) or (st.get("card") or {}).get("kind") in
            ("approval", "escalation", "cluster", "drift", "stop_loss", "platform", "quarantine_lift")) else "ok"
    segs = st.get("segments") or []
    return {"mode": "stream", "enabled": bool(cfg.get("enabled")), "alarm": alarm,
            "paused_reason": cfg.get("paused_reason") or (st.get("paused") or {}).get("reason"),
            "goal": cfg.get("goal"), "autonomy": cfg.get("autonomy"), "data_autonomy": cfg.get("data_autonomy"),
            "domain": cfg.get("domain"), "sid": (cfg.get("stream") or {}).get("sid"),
            "lanes": {ln: {"phase": (lanes.get(ln) or {}).get("phase"),
                           "item": ((lanes.get(ln) or {}).get("item") or {}).get("lever"),
                           "status": ((lanes.get(ln) or {}).get("item") or {}).get("status"),
                           "hold": (lanes.get(ln) or {}).get("hold") or (lanes.get(ln) or {}).get("diag_hold")}
                      for ln in ALL_LANES if ln in lanes or ln in LANES},
            "segments": len(segs), "committed": sum(1 for s in segs if s.get("committed")),
            "milestones": len(st.get("milestones") or []), "card": st.get("card"),
            "cards": (st.get("cards") or [])[-10:], "errors": st.get("errors"), "drift": st.get("drift"),
            "stage_a": st.get("stage_a"), "stage_c": st.get("stage_c"), "capacity": (st.get("capacity") or {}).get("chosen"),
            "complete_available": complete_available(st), "completed_by": cfg.get("completed_by"),
            "last_tick_utc": st.get("last_tick_utc"), "last_snapshot_utc": st.get("last_snapshot_utc"),
            "ticks": st.get("ticks")}


def complete_available(st):
    """6.7: COMPLETE is available to a person (never taken by the ticker) when
    D29 has moved DATA to WAIT_DATA, nothing is in flight in any lane, and every
    committed segment's increments have been through a milestone."""
    st = st or {}
    lanes = st.get("lanes") or {}
    if (lanes.get("DATA") or {}).get("phase") != "WAIT_DATA":
        return False
    if any((lanes.get(ln) or {}).get("item") for ln in ALL_LANES):
        return False
    segs = [s for s in st.get("segments") or [] if s.get("committed")]
    ms = [m for m in st.get("milestones") or [] if m.get("done")]
    last = max([m.get("after_segment") or 0 for m in ms] or [0])
    return len(segs) <= last


def release(name, by, cfg_hooks=None, lab_repo=None, clock=None):
    """A person releases the lanes of stream campaign `name` that a stop-loss
    held (6.6): the campaign is enabled again with a resume stamp
    (configure_stream enable), and the next tick frees every lane held before
    that stamp (StreamRun._resumed: 'lane_released' in the campaign ledger)
    and clears its steps' failure counts. Returns the lanes it will free."""
    from . import campaign as C
    cfg_hooks = cfg_hooks or C.default_cfg_hooks()
    raw = ((cfg_hooks[0]() or {}).get(C.CONFIG_KEY) or {}).get(name)
    if not isinstance(raw, dict) or raw.get("mode") != "stream":
        raise ValueError("%r is not a stream campaign" % (name,))
    c = stream_config(raw, name)
    st = load_state(StreamPaths(lab_repo, M.campaign_domain(c)), name) or {}
    held = {ln: (st.get("lanes") or {}).get(ln, {}).get("hold") for ln in LANES
            if ((st.get("lanes") or {}).get(ln) or {}).get("hold")}
    out = configure_stream(name, by, enable=True, cfg_hooks=cfg_hooks, lab_repo=lab_repo, clock=clock)
    return {"ok": True, "campaign": name, "released_by": by, "resumed_utc": out.get("resumed_utc"),
            "held_lanes": held}


def status(name=None, cfg_hooks=None, lab_repo=None):
    from . import campaign as C
    cfg_hooks = cfg_hooks or C.default_cfg_hooks()
    camps = (cfg_hooks[0]() or {}).get(C.CONFIG_KEY) or {}
    out = {}
    for n in sorted(camps):
        raw = camps.get(n)
        if (name and n != name) or not isinstance(raw, dict) or raw.get("mode") != "stream":
            continue
        c = stream_config(raw, n)
        paths = StreamPaths(lab_repo, M.campaign_domain(c))
        out[n] = {"config": c, "summary": summary(c, load_state(paths, n))}
    return out


# ------------------------------------------------------------------ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(prog="inc_autopilot.stream", description="stream-mode campaigns "
                                 "(docs/CONTINUOUS_LOOP.md 6)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--lab-repo", default=None)
    sub = ap.add_subparsers(dest="cmd")

    def cap_or_none(v):
        # a time-based cap takes a number or `none` (CLEAR: no cap, the 2026-10-04 amendment)
        if str(v).strip().lower() == CLEAR:
            return CLEAR
        try:
            return float(v)
        except ValueError:
            raise argparse.ArgumentTypeError("a number or %r, not %r" % (CLEAR, v))

    def settings(p):
        p.add_argument("--name", required=True)
        p.add_argument("--by", required=True)
        p.add_argument("--domain", default=None)
        p.add_argument("--autonomy", choices=("off", "envelope"), default=None)
        p.add_argument("--data-autonomy", choices=("off", "on"), default=None)
        for f in ("--envelope-su", "--window-cap-su", "--daily-cap-su", "--alloc-reserve-su", "--collect-gb-envelope",
                  "--collect-gb-daily"):
            clearable = f[2:].replace("-", "_") in CLEARABLE
            p.add_argument(f, type=cap_or_none if clearable else float, default=None,
                           help="a number, or none for no cap" if clearable else None)
        p.add_argument("--envelope-end-utc", default=None)
        p.add_argument("--protocol-v3-accepted", action="store_true")
    settings(sub.add_parser("enable"))
    settings(sub.add_parser("configure", help="change settings only: no enable, no release of a pause or a hold"))
    s = sub.add_parser("status")
    s.add_argument("--name", default=None)
    rl = sub.add_parser("release")
    rl.add_argument("--name", required=True)
    rl.add_argument("--by", required=True)
    ro = sub.add_parser("reopen", help="a person reopens a closed source (the next tick applies it)")
    ro.add_argument("--name", required=True)
    ro.add_argument("--source", required=True)
    ro.add_argument("--by", required=True)
    ro.add_argument("--why", required=True)
    c = sub.add_parser("complete")
    c.add_argument("--name", required=True)
    c.add_argument("--by", required=True)
    r = sub.add_parser("lab-run")
    r.add_argument("--spec", required=True)
    y = sub.add_parser("lab-sync")
    y.add_argument("--lab-inc", required=True)
    y.add_argument("--source", required=True)
    y.add_argument("--file", default=None)
    y.add_argument("--target", default="")
    y.add_argument("--data-target", default="")
    y.add_argument("--names", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "lab-run":
        return lab_run(a.spec)
    if a.cmd == "lab-sync":
        res = lab_sync(a.lab_inc, a.source, a.target, a.data_target or None, file=a.file, names=a.names)
        print(json.dumps(res, sort_keys=True))
        return 0 if res.get("ok") else 1
    from . import campaign as C
    hooks = C.default_cfg_hooks(a.config)
    try:
        if a.cmd in ("enable", "configure"):
            out = configure_stream(a.name, a.by, domain=a.domain, enable=True if a.cmd == "enable" else None,
                                   autonomy=a.autonomy,
                                   data_autonomy=a.data_autonomy, envelope_su=a.envelope_su,
                                   window_cap_su=a.window_cap_su, daily_cap_su=a.daily_cap_su,
                                   alloc_reserve_su=a.alloc_reserve_su, collect_gb_envelope=a.collect_gb_envelope,
                                   collect_gb_daily=a.collect_gb_daily, envelope_end_utc=a.envelope_end_utc,
                                   protocol_v3_accepted=a.protocol_v3_accepted or None, cfg_hooks=hooks,
                                   lab_repo=a.lab_repo)
        elif a.cmd == "release":
            out = release(a.name, a.by, cfg_hooks=hooks, lab_repo=a.lab_repo)
        elif a.cmd == "reopen":
            full = configure_stream(a.name, a.by, cfg_hooks=hooks, lab_repo=a.lab_repo, reopen_source=a.source,
                                    reopen_why=a.why)
            out = {"ok": True, "campaign": a.name, "reopened": a.source,
                   "stamp": (full.get("reopened_sources") or {}).get(a.source)}
        elif a.cmd == "status":
            out = status(a.name, hooks, a.lab_repo)
        elif a.cmd == "complete":
            st = status(a.name, hooks, a.lab_repo).get(a.name) or {}
            if not (st.get("summary") or {}).get("complete_available"):
                raise ValueError("COMPLETE is not available: 6.7's conditions do not all hold")
            out = configure_stream(a.name, a.by, complete=True, enable=False, cfg_hooks=hooks, lab_repo=a.lab_repo)
        else:
            ap.print_help(sys.stderr)
            return 2
    except ValueError as err:
        print(json.dumps({"ok": False, "error": str(err)}))
        return 1
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
