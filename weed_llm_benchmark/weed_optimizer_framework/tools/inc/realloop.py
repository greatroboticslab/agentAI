"""INC Steps 2-3, the incremental loop on real data: the experiment builder
(docs/INCREMENTAL_PROTOCOL.md, "Steps 2-3"; docs/INCREMENTAL_PROTOCOL_RUNNER.md,
"Real-loop build").

    python -m weed_optimizer_framework.tools.inc.realloop build --exp real_v1
        --replay-mode {sample,full} --recipes full[,freeze,lora] [--gate-flips-mode {negative,net}]
        [--base INC_DIR/step1/base_B.jsonl] [--n-verified 6] [--size M] [--no-truth]
        [--increment-sources relevance] [--relevance INC_DIR/step1/relevance.json]
        | --increment-sources evidence [--min-evidence 1]
        | --increment-sources recovered --step1-overlay INC_DIR/step1_r1 --size M
        [--testing | --testing-settings JSON] [--quiet]

--increment-sources recovered is a separate protocol version (the funnel
audit's realloop_v2, docs/FUNNEL_AUDIT.md section 9; runner
docs/FUNNEL_AUDIT_RUNNER.md 5.6.3); section "Recovered increments" at the end
of this docstring describes it. Everything above that section describes the
'relevance' and 'evidence' builds, which it leaves byte for byte as they were.

It builds a 'chain' experiment the way inc/pilot.py builds the pilot: the
manifests under INC_DIR/<exp>/manifests/, build_summary.json and the
definition, then driver init (exp.json, state.json, the base runs and the
truth arm submitted). An experiment is built once; a build that stopped before
driver init is rebuilt from scratch (pilot._check_new).

Inputs (Step 1), one consistent set or the build refuses:
  * inc/select.py's build in the directory of --base: select_summary.json,
    base_B.jsonl, increment_pool.jsonl, select_clusters.csv. --base must be
    the base_B.jsonl the summary names (sha256): the increment pool is the
    rest of the verified set only relative to that base;
  * inc/verify.py's pool.jsonl, pool_meta.jsonl, verified.jsonl, conflicts.csv
    and admit_summary.json (verify's own paths). select's summary must name
    this verified.jsonl and pool_meta.jsonl; admit_summary.json must name this
    verified.jsonl, pool.jsonl and pool_meta.jsonl;
  * every pool image's verdict is rebuilt from the files (admitted: in
    verified.jsonl, as the same row; conflict: a box in conflicts.csv;
    unknown: the rest) and must give admit_summary.json's image verdict
    counts for every source;
  * the select build read the train_core LOCK.json records and the
    never-train index in place now (pilot.select_provenance): a base
    selected under an earlier lock is stale. A production build refuses it,
    a testing build warns.

Base: --base, copied into the experiment as manifests/base_B.jsonl (same
bytes) and checked by pilot.check_training_manifest (inc/train.py's manifest
check and the never-train guard, fail closed); cold with the protocol's
recipe, 3 seeds. These are the same three runs as base_b_v1 (inc.pilot
build-baseline on the same base_B.jsonl), and the loop's final runs read B's
test again; the driver cannot import another experiment's base weights. The
build records every baseline experiment already built on the same manifest
bytes (build_summary.json "same_base_baselines").

Increments, M images each. M = --size, by default INC_FRAC (10 %) of the
base's images, rounded (the protocol's size, and select's default):
  * V1..VN (--n-verified, default 6) and OTHER_HEAVY:
    select.increments(exp, N, M, other_heavy=True, relevance=R) as it stands
    (with --increment-sources evidence: sources='evidence' instead of R):
    cluster-balanced, disjoint, whole near-dup groups, the never-train guard,
    seeded by stable_int(exp) (OTHER_HEAVY by stable_int(exp + "/otherplant")).
    They are select's inc_01.jsonl .. inc_NN.jsonl and inc_otherplant.jsonl,
    written in the experiment's manifests/ with increments_summary.json.
  * --increment-sources picks the relevance criterion of those draws:
    'relevance' (the default: R below, as before) or 'evidence' (source-level
    species evidence, select's --sources evidence; docs/INCREMENTAL_PROTOCOL.md,
    Steps 2-3): only sources in which verify admit judged at least
    --min-evidence (select.MIN_EVIDENCE = 1) cwd12-species boxes 'verified'
    (verify's admit_summary.json, the one checked below) are drawn from. With
    'evidence' no relevance file is read, and an explicit --relevance refuses;
    a production build needs none. Before anything is written, the build
    checks the evidence (select.load_evidence: admit_summary.json and
    select_summary.json agree) and the evidenced pool's capacity
    (select.pool_capacity): it refuses unless the pool holds (N + 1) * M images
    and M of them in OtherPlant-heavy near-dup groups. An 'evidence' build
    records its mode in exp.json's and build_summary.json's step1 block
    ("increment_sources": mode, min_evidence, rule); a 'relevance' build
    records no such key (a missing key means 'relevance'), so its definition
    is the one built before --increment-sources existed and an experiment
    built then can still be re-inited. With 'evidence', build_summary.json
    "evidence" records the evidenced and excluded sources, the cross-check and
    the capacity, and "unverified" records UNVERIFIED's source's verified
    boxes.
  * The relevance filter R (inc/relevance.py; --increment-sources relevance,
    the default): the verifier certifies species
    labels, not relevance, so the increment pool holds whole sources that are
    not plant photographs. R is --relevance, by default relevance.json next to
    --base (INC_DIR/step1/relevance.json) when it exists. Every increment-pool
    image of a source R does not pass leaves the V1..VN / OTHER_HEAVY draw.
    A production build refuses without R; a testing build without it warns
    and draws unfiltered. Any build refuses, before anything is written, an R
    that relevance.load refuses (another select build, a failed calibration
    check, a status that does not follow from its numbers). UNVERIFIED is never filtered: it is the planted,
    realistic bad increment. R's sha256 is in exp.json's step1 files, and
    build_summary.json ("relevance") records the excluded sources and images;
    "unverified" records the relevance status of UNVERIFIED's source.
  * UNVERIFIED (manifests/inc_unverified.jsonl): M images of ONE harvested
    source that verify did not admit (image verdict conflict or unknown).
      - Every non-admitted pool image must have a dHash in pool_meta.jsonl,
        clear the never-train guard, come from no never-train dataset and
        share no key, image path or image bytes with a train_core image of
        the base (select's own checks: guard_rows, check_pool_against_core);
        nor may it have the key, image path or image sha256 of a harvested
        image of the base or of V1..VN / OTHER_HEAVY (verify's pool holds no
        exact duplicates and no train_core copies). Any failure stops the
        build, since it means the Step 1 files are stale.
      - An image is eligible unless it lies within NEAR_DUP_BITS dHash bits
        of an image of the base or of V1..VN / OTHER_HEAVY (base dHashes from
        the base check, increment dHashes from pool_meta.jsonl).
      - The source is the one with the most eligible images among those with
        at least M (ties: the lower name). With no such source the build
        refuses.
      - M images are drawn with numpy.random.default_rng(stable_int(exp +
        "/unverified")) without replacement over the source's eligible images
        sorted by key. The rows are pool.jsonl's own, with the labels the
        species join gave them, never edited.
      - build_summary.json ("unverified") records the source, the
        per-source candidate table, the near-dup exclusions (first 20, with
        the image each one is near), the drawn images' verdicts, their
        conflict boxes (label -> predicted species) and the source's verdict
        counts from admit_summary.json; exp.json's step records the source,
        the verdicts and the rule.
  * Every increment then passes pilot.check_training_manifest. The base and
    the increments are checked pairwise disjoint (key, image path, sha256).

Sequence: the first two verified increments, UNVERIFIED, the third verified
increment, OTHER_HEAVY, then the rest. For N = 6 that is V1, V2, UNVERIFIED,
V3, OTHER_HEAVY, V4, V5, V6. Clean = every verified increment (V1..VN and
OTHER_HEAVY). So the truth arm's T_k grows through them and T_final = the base
plus every verified increment, while UNVERIFIED is decided (every chain and the
truth arm) but never joins T_k.

Recipes: --recipes names a subset of pilot.inc_recipes() (the pilot's
definitions, not redefined here); base and truth runs use pilot.cold_recipe().
--replay-mode goes into exp.json as the pilot's does, and so does
--gate-flips-mode (default negative, protocol v1; 'net' is protocol v2), as
exp.json's and build_summary.json's gate block {"flips_mode": ...}. The truth
arm is on unless --no-truth. Attribution: the driver runs protocol steps 1-3.
Step 4's evidence is Step 1 itself (every box of a verified increment was
verified, and UNVERIFIED's verdicts are in the summary). Step 5
(leave-one-source-out runs) is not run; exp.json and every gate entry record
that.

Records. exp.json (the definition the driver runs): the base and every step
with its manifest, sha256, n_images, clean flag and kind (UNVERIFIED also its
source, verdicts and rule), M, the recipes, the replay mode, the gate block,
the truth arm, the Step 1 files with their sha256 (select's train_core and
never-train index among them; with --increment-sources evidence, the mode),
the effective warmup and the attribution scope.
build_summary.json, in addition: per-source and per-class counts of the base
and of every increment, select's draw parameters and per-increment table, the
unverified section, the never-train lock status, the select provenance record
and the baseline experiments already built on the same base.

A production build needs LOCK.json and the never-train index it recorded
(pilot.never_train_status), a select build that read that lock's
train_core and that index (pilot.select_provenance), and the relevance file
(with --increment-sources relevance); a testing build only warns.

Recovered increments (--increment-sources recovered; realloop_v2).
The base is B as above (the same checks: select's base_B, Step 1 consistent,
the select provenance, pilot.check_training_manifest). The increments come
from the funnel audit's recovery overlay, INC_DIR/step1_r1/ (--step1-overlay),
written by `funnel recover`: recovery.json (funnel-recovery/1) and
recovered_pool.jsonl, one row per recovered image (the manifest keys plus
pool, policy, unmasked_image, unmasked_sha256, dhash_unmasked, near_dup3 and
the rest of the runner's section 4.18). Source labels and step1/labels are
never read as increment labels here; a row's label is the overlay's.
  * Flags. --increment-sources recovered needs --step1-overlay and --size (M,
    the requested images per step) and keeps the truth arm on; it refuses
    --relevance, --min-evidence, --n-verified and --no-truth. --step1-overlay
    without the mode refuses.
  * load_overlay refuses, before anything is written: a recovery.json that is
    not funnel-recovery/1 or not 'complete'; one made by a testing recover for
    a production build; an input it recorded (header "inputs") that is missing
    or no longer hashes as recorded; a guard record with a never-train hit, an
    unhashable image or an H6 copy, or a changed source label; a guard record
    that does not cover every row (never-train and H6 each checked every
    unmasked original and every masked copy of recovered_pool.jsonl); no H10d
    domain-dev record, or one whose files do not hash as recorded
    (pilot.domain_dev_record: the record names the rows, or the
    domain_dev.json document that names them); a
    recovered_pool.jsonl whose sha256 is not the one recovery.json records; a
    row without the overlay keys or without a near_dup3 group, of an unknown
    pool, doubled, of a quarantined source, or held out as domain dev; a
    domain-dev key that is not a pool image; a row of a drawn pool (VETO, AUTH,
    CLASS, JUDGE) that names no stratum, or a stratum whose gate recovery.json
    does not record as passed; a row that is in the base (key, image
    path or image sha256, the unmasked original's sha256 included); a pool
    row whose unmasked original is not verify's pool image (path, sha256,
    source), and any row whose unmasked original does not hash as recorded;
    a row whose unmasked original fails the never-train guard, through its
    pool_meta.jsonl dHash (the dHash recorded in the row must be that one; a
    row outside the pool is hashed from its file); rows within 3 dHash bits of
    each other with different near_dup3 groups; and a row whose image or
    label fails pilot.check_training_manifest (inc/train.py's manifest check:
    label and image bytes as recorded, the INC class space, and the
    never-train guard on the image that is trained, the masked copy).
  * Near-duplicate exclusion (overlay_exclusions). As in the other modes, no
    increment replays a photograph that is in the base or in another
    increment (select's near-duplicate groups; UNVERIFIED's exclusion), and
    none replays an H10d domain-dev photograph. A near_dup3 group is left out
    of every draw when one of its rows lies within NEAR_DUP_BITS of a base
    image of verify's pool (pool_meta.jsonl dHash; the base's train_core
    images are more than HOLDOUT_NEAR_DUP_BITS from every pool image, verify's
    cwd12_copy stage) or of a domain-dev image, or when its rows lie in more
    than one pool (drawing each pool's part would put one photograph in two
    increments). The groups left out, with their reasons, pools and sizes,
    are recorded in build_summary.json "recovered" "excluded".
  * Pools. VETO, AUTH, CLASS and JUDGE (a FETCH row is checked, never drawn).
    A pool's rows are grouped by near_dup3 (each group's rows sorted by key,
    groups by their first key).
  * Plan (plan_recovered). RECOVERED_SEQUENCE is REC-VETO, REC-AUTH-1,
    REC-CLASS-1, REC-JUDGE-1, REC-AUTH-2, REC-CLASS-2. floor = 0.05 x |B|
    (a real number). An arm whose capacity per step (its images / the steps
    drawn from it) is below the floor is dropped and reported. M_eff =
    min(M, the smallest capacity per step over the kept arms, rounded down);
    below the floor the build refuses with the numbers. Each step of a
    dropped arm is drawn from the kept arm with the most remaining capacity
    among SUBSTITUTION_ORDER (AUTH, CLASS, VETO; a tie goes to the earlier),
    in sequence order; with none holding M_eff images the build refuses.
  * Draw (draw_recovered). Whole near_dup3 groups of the step's pool that no
    earlier step took, sorted by first key and permuted with
    numpy.random.default_rng(stable_int(exp + "/" + step)); a group is added
    while it fits M_eff exactly and skipped when it would overshoot; the draw
    refuses if it cannot reach M_eff. The manifest rows are the overlay rows'
    manifest keys (manifests/<step>.jsonl).
  * Steps: clean false, kind 'recovered', pool, policy_counts and
    substituted_from. There is no UNVERIFIED and no OTHER_HEAVY; T_k never
    grows, so every truth step compares B + D_k with B's own runs.
  * Records. exp.json step1.increment_sources = {mode 'recovered', overlay
    {dir, recovery_sha256, recovered_pool_sha256}, rule, m_requested, m,
    dropped, substituted}; build_summary.json "recovered" holds the plan with
    the capacities, per-step counts by pool, policy and class, the overlay's
    checks and guard record, and the near_dup3 groups left out of the draw
    ("excluded"; those that span two pools are also counted under
    "cross_pool_near_dup3").
  * Every step passes pilot.check_training_manifest and the base and the
    steps are pairwise disjoint, as in the other modes.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import shutil
import sys
from pathlib import Path

from . import common as C
from . import driver as D
from . import pilot as P
from . import select as S
from ..near_dup import NEAR_DUP_BITS, NearHashIndex
from .scorer import TEST_ENV

N_VERIFIED = 6
BASE_NAME = "base_B"
UNVERIFIED = "UNVERIFIED"
OTHER_HEAVY = "OTHER_HEAVY"
UNVERIFIED_MANIFEST = "inc_unverified.jsonl"
VERIFIED_BEFORE_UNVERIFIED = 2       # verified increments decided before UNVERIFIED
VERIFIED_BEFORE_OTHER_HEAVY = 3      # ... and before OTHER_HEAVY
MIN_DECIDED = 6                      # the protocol's acceptance: at least 6 increments decided
BUILDER = "inc.realloop build"
KIND_VERIFIED, KIND_OTHER_HEAVY, KIND_UNVERIFIED = "verified", "otherplant_heavy", "unverified"
UNVERIFIED_RULE = ("M images of the one harvested source with the most eligible images verify did not "
                   "admit (conflict or unknown), among sources with at least M (ties: the lower name); "
                   "eligible = not within %d dHash bits of an image of the base or of a verified increment "
                   "(a never-train hit, a train_core copy or the same image as one of those stops the "
                   "build); drawn with default_rng(stable_int(exp + '/unverified')) over the eligible "
                   "images sorted by key; labels as the species join gave them" % NEAR_DUP_BITS)
ATTRIBUTION_SCOPE = {
    "run": ["1 recipe vs data", "2 classification vs localisation", "3 per-species deltas"],
    "not_run": {"4": "label audit: not re-run per step; the Step 1 verifier judged every box already "
                     "(every species box of a verified increment is verified; the unverified increment's "
                     "verdicts are in build_summary.json 'unverified')",
                "5": "leave-one-source-out cand runs for a multi-source increment: not run by the driver; "
                     "each increment's per-source counts are in build_summary.json"},
}

# --increment-sources recovered (realloop_v2; module docstring, "Recovered increments")
SOURCES_RECOVERED = "recovered"
INCREMENT_SOURCE_MODES = S.SOURCE_MODES + (SOURCES_RECOVERED,)
RECOVERED_SEQUENCE = ("REC-VETO", "REC-AUTH-1", "REC-CLASS-1", "REC-JUDGE-1", "REC-AUTH-2", "REC-CLASS-2")
RECOVERED_POOLS = {"REC-VETO": "VETO", "REC-AUTH-1": "AUTH", "REC-AUTH-2": "AUTH", "REC-CLASS-1": "CLASS",
                   "REC-CLASS-2": "CLASS", "REC-JUDGE-1": "JUDGE"}
SUBSTITUTION_ORDER = ("AUTH", "CLASS", "VETO")
OVERLAY_POOLS = ("VETO", "AUTH", "CLASS", "JUDGE", "FETCH")      # recovered_pool.jsonl "pool" values
RECOVERED_FLOOR_FRAC = 0.05                                     # floor of M: 5 % of |B|
NEAR_DUP3_BITS = 3                                              # the overlay's near_dup3 grouping
KIND_RECOVERED = "recovered"
RECOVERY_FILE = "recovery.json"
RECOVERED_POOL_FILE = "recovered_pool.jsonl"
RECOVERY_FORMAT = "funnel-recovery/1"
RECOVERY_COMPLETE = "complete"
OVERLAY_ROW_KEYS = C.MANIFEST_KEYS + ("pool", "policy", "strata", "unmasked_image", "unmasked_sha256",
                                      "dhash_unmasked", "near_dup3")
DRAWN_POOLS = tuple(sorted(set(RECOVERED_POOLS.values())))       # the pools a step is drawn from
EXCLUDE_NEAR_BASE, EXCLUDE_NEAR_DOMAIN_DEV, EXCLUDE_CROSS_POOL = "near_base", "near_domain_dev", "cross_pool"
RECOVERED_RULE = ("M images per step, whole near-duplicate groups (the overlay's 3-bit near_dup3) of the step's pool "
                  "that no earlier step took, leaving out every group with a row within %d dHash bits of a base "
                  "image or of an H10d domain-dev image, and every group whose rows lie in more than one pool; the "
                  "groups sorted by first key and permuted with default_rng(stable_int(exp + '/' " % NEAR_DUP_BITS +
                  "+ step)); a group is added while it fits M exactly and skipped when it would overshoot, and the "
                  "draw refuses if it cannot reach M. M = min(--size, the smallest capacity per step (images / steps "
                  "drawn from the arm) over the arms at or above the floor 0.05 x |B|); an arm below the floor is "
                  "dropped, and each of its steps is drawn from the kept arm with the most remaining capacity among "
                  "AUTH, CLASS, VETO (a tie goes to the earlier); every step is unclean and the truth arm compares "
                  "B + D_k with B")
RECOVERED_ATTRIBUTION_SCOPE = {
    "run": ["1 recipe vs data", "2 classification vs localisation", "3 per-species deltas"],
    "not_run": {"4": "label audit: not re-run per step; the funnel audit measured the recovered labels' precision "
                     "before recovery (step1_r1/recovery.json gates, funnel/audit_v1.json)",
                "5": "leave-one-source-out cand runs for a multi-source increment: not run by the driver; "
                     "each increment's per-source counts are in build_summary.json"},
}


class RealLoopError(P.PilotError):
    """A condition under which the Steps 2-3 experiment must not be built."""


def log(msg):
    print("[inc.realloop] %s" % msg, flush=True)


# ------------------------------------------------------------ the plan
def verified_name(j):
    return "V%d" % j


def sequence(n_verified):
    """Step names in order: the first VERIFIED_BEFORE_UNVERIFIED verified
    increments, UNVERIFIED, verified ones up to VERIFIED_BEFORE_OTHER_HEAVY,
    OTHER_HEAVY, the rest."""
    n = int(n_verified)
    if n < 1:
        raise RealLoopError("--n-verified must be >= 1, got %r" % n_verified)
    v = [verified_name(j) for j in range(1, n + 1)]
    a, b = VERIFIED_BEFORE_UNVERIFIED, VERIFIED_BEFORE_OTHER_HEAVY
    return v[:a] + [UNVERIFIED] + v[a:b] + [OTHER_HEAVY] + v[b:]


def parse_recipes(recipes):
    """Recipe names (a list, or 'full,freeze,lora'), each one of the pilot's."""
    names = ([x.strip() for x in recipes.split(",") if x.strip()] if isinstance(recipes, str)
             else list(recipes or []))
    table = P.inc_recipes()
    unknown = [r for r in names if r not in table]
    if not names or unknown or len(set(names)) != len(names):
        raise RealLoopError("--recipes must name distinct recipes among %s, got %r"
                            % (sorted(table), recipes))
    return names


def default_size(n_base):
    """The protocol's increment size: INC_FRAC of the base's images (select's default)."""
    return max(1, int(round(S.INC_FRAC * n_base)))


def choose_source(eligible, m):
    """The source with the most eligible images among those with at least m
    (ties: the lower name)."""
    ok = [(-n, s) for s, n in eligible.items() if n >= m]
    if not ok:
        best = max(eligible.items(), key=lambda kv: (kv[1], kv[0]), default=(None, 0))
        raise RealLoopError("no harvested source has %d eligible images that verify did not admit "
                            "(the most: %s with %d)" % (m, best[0], best[1]))
    return min(ok)[1]


# ------------------------------------------------------------- Step 1
def _sha(path):
    return C.sha256_file(path) if Path(path).is_file() else None


def load_step1(base_path):
    """Step 1's files as one checked set (module docstring, 'Inputs')."""
    V = S._verify()
    base_path = Path(os.path.abspath(str(base_path)))
    base_dir = base_path.parent
    if not base_path.is_file():
        raise RealLoopError("no base manifest at %s (run inc.select build)" % base_path)
    sel = S._read_json(base_dir / S.SUMMARY)
    if sel is None:
        raise RealLoopError("no %s next to %s: the base must be the base_B.jsonl of an inc.select build"
                            % (S.SUMMARY, base_path))
    if not S._outputs_match(sel, base_dir, (S.BASE_B, S.POOL, S.CLUSTERS)):
        raise RealLoopError("%s, %s or %s in %s changed since the select build that wrote %s"
                            % (S.BASE_B, S.POOL, S.CLUSTERS, base_dir, S.SUMMARY))
    base_sha = C.sha256_file(base_path)
    if base_sha != sel["outputs"][S.BASE_B]["sha256"]:
        raise RealLoopError("%s is not the %s the select build wrote (%s != %s); the increment pool "
                            "is the rest of the verified set only relative to that base"
                            % (base_path, S.BASE_B, base_sha[:12], sel["outputs"][S.BASE_B]["sha256"][:12]))
    inputs = sel.get("inputs") or {}
    shas = {"verified": _sha(V.VERIFIED_MANIFEST), "pool_meta": _sha(V.POOL_META), "pool": _sha(V.POOL)}
    for name, path in (("verified", V.VERIFIED_MANIFEST), ("pool_meta", V.POOL_META)):
        if shas[name] is None or shas[name] != (inputs.get(name) or {}).get("sha256"):
            raise RealLoopError("verify's %s (%s) is not the one the select build read"
                                % (Path(path).name, path))
    for path in (V.POOL, V.CONFLICTS, V.ADMIT_SUMMARY):
        if not Path(path).is_file():
            raise RealLoopError("missing %s (run inc.verify admit)" % path)
    admit = S._read_json(V.ADMIT_SUMMARY)
    if not isinstance(admit, dict):
        raise RealLoopError("%s cannot be read" % V.ADMIT_SUMMARY)
    ain = admit.get("inputs") or {}
    if (admit.get("verified_sha256") != shas["verified"] or ain.get("pool_sha256") != shas["pool"]
            or ain.get("pool_meta_sha256") != shas["pool_meta"]):
        raise RealLoopError("%s does not name this verified.jsonl, pool.jsonl and pool_meta.jsonl: they "
                            "changed after verify admit" % V.ADMIT_SUMMARY)
    pool = C.read_manifest(V.POOL)
    verified = C.read_manifest(V.VERIFIED_MANIFEST)
    with open(V.CONFLICTS, newline="") as fh:
        conflicts = list(csv.DictReader(fh))
    verdict, per_source = image_verdicts(pool, verified, conflicts, admit)
    return {"dir": base_dir, "base": base_path, "base_sha256": base_sha, "select": sel,
            "select_summary": {"path": str(base_dir / S.SUMMARY), "sha256": C.sha256_file(base_dir / S.SUMMARY)},
            "admit": admit, "pool": pool, "conflicts": conflicts, "verdict": verdict,
            "per_source": per_source,
            "files": {"pool": {"path": str(V.POOL), "sha256": shas["pool"]},
                      "pool_meta": {"path": str(V.POOL_META), "sha256": shas["pool_meta"]},
                      "verified": {"path": str(V.VERIFIED_MANIFEST), "sha256": shas["verified"]},
                      "conflicts": {"path": str(V.CONFLICTS), "sha256": C.sha256_file(V.CONFLICTS)},
                      "admit_summary": {"path": str(V.ADMIT_SUMMARY),
                                        "sha256": C.sha256_file(V.ADMIT_SUMMARY)}}}


def image_verdicts(pool, verified, conflicts, admit):
    """({key: image verdict}, {source: Counter}) rebuilt from verify's files:
    admitted = in verified.jsonl (as the same row), conflict = a box in
    conflicts.csv, unknown = the rest. Raises unless the per-source counts
    are admit_summary.json's."""
    V = S._verify()
    by_key = {r["key"]: r for r in pool}
    if len(by_key) != len(pool):
        raise RealLoopError("%s holds duplicate keys" % V.POOL)
    ver = {r["key"]: r for r in verified}
    odd = [k for k, r in ver.items()
           if k not in by_key or any(r.get(f) != by_key[k].get(f) for f in C.MANIFEST_KEYS)]
    if odd:
        raise RealLoopError("%d verified row(s) are not pool rows, e.g. %s" % (len(odd), odd[:3]))
    ckeys = {c.get("key") for c in conflicts}
    odd = sorted(k for k in ckeys if k not in by_key or k in ver)
    if odd:
        raise RealLoopError("%d conflict image(s) are not pool images or were admitted, e.g. %s"
                            % (len(odd), odd[:3]))
    verdict = {k: V.ADMITTED if k in ver else V.CONFLICT if k in ckeys else V.UNKNOWN for k in by_key}
    per = collections.defaultdict(collections.Counter)
    for k, r in by_key.items():
        per[r["source"]][verdict[k]] += 1
    got = {s: V._counter(c) for s, c in per.items()}
    want = {s: (d or {}).get("images") for s, d in (admit.get("per_slug") or {}).items()}
    if got != want:
        diff = sorted(s for s in set(got) | set(want) if got.get(s) != want.get(s))
        raise RealLoopError("the image verdicts rebuilt from %s, %s and %s differ from %s for %d "
                            "source(s), e.g. %s: rebuilt %s, admit %s"
                            % (Path(V.POOL).name, Path(V.VERIFIED_MANIFEST).name, Path(V.CONFLICTS).name,
                               Path(V.ADMIT_SUMMARY).name, len(diff), diff[0], got.get(diff[0]),
                               want.get(diff[0])))
    return verdict, per


# -------------------------------------------------- the unverified increment
def draw_unverified(exp, m, step1, taken, guard):
    """(rows, info): the UNVERIFIED increment (module docstring). taken is
    [(name, rows, {key: dHash})] of the base (named BASE_NAME) and the
    verified increments; the base's rows that are not pool images are its
    train_core part."""
    import numpy as np
    V = S._verify()
    verdict = step1["verdict"]
    cands = sorted((r for r in step1["pool"] if verdict[r["key"]] != V.ADMITTED), key=lambda r: r["key"])
    if not cands:
        raise RealLoopError("verify admitted every pool image; there is no unverified data to draw from")
    pool_keys = {r["key"] for r in step1["pool"]}
    core = [r for name, rows, _h in taken if name == BASE_NAME for r in rows if r["key"] not in pool_keys]
    try:
        dh = S.read_pool_dhash(V.POOL_META, [r["key"] for r in cands])
        S.guard_rows(guard, cands, dh, "the pool images verify did not admit")
    except S.SelectError as e:
        raise RealLoopError(str(e))
    try:
        S.check_pool_against_core(cands, core)
    except S.SelectError as e:
        raise RealLoopError("among the pool images verify did not admit: %s" % e)
    owner = {}
    index = NearHashIndex()
    for name, rows, hashes in taken:
        for r in rows:
            for f in ("key", "image", "sha256"):
                owner.setdefault((f, r[f]), "%s/%s" % (name, r["key"]))
            index.add(int(hashes[r["key"]]), "%s/%s" % (name, r["key"]), max_bits=NEAR_DUP_BITS)
    same = []
    for r in cands:
        hit = next(((f, owner[(f, r[f])]) for f in ("key", "image", "sha256") if (f, r[f]) in owner), None)
        if hit is not None:
            same.append((r["key"],) + hit)
    if same:
        raise RealLoopError("%d image(s) verify did not admit share a key, image path or image sha256 with "
                            "an image of the base or of a verified increment, e.g. %s: verify's pool holds "
                            "no exact duplicates, so the Step 1 files are stale" % (len(same), same[:3]))
    table = collections.defaultdict(collections.Counter)
    eligible = collections.defaultdict(list)
    near = []
    for r in cands:
        t = table[r["source"]]
        t["non_admitted"] += 1
        t[verdict[r["key"]]] += 1
        hit = index.find(int(dh[r["key"]]))
        if hit is not None:
            t["excluded_near_dup"] += 1
            near.append({"key": r["key"], "source": r["source"], "near": hit[0], "bits": hit[1]})
            continue
        t["eligible"] += 1
        eligible[r["source"]].append(r)
    source = choose_source({s: t["eligible"] for s, t in table.items()}, m)
    pool = eligible[source]
    rng = np.random.default_rng(C.stable_int(exp + "/unverified"))
    picked = [pool[i] for i in sorted(int(x) for x in rng.choice(len(pool), size=m, replace=False))]
    rows = [{k: r[k] for k in C.MANIFEST_KEYS} for r in picked]
    drawn = {r["key"] for r in rows}
    cbox = [c for c in step1["conflicts"] if c.get("key") in drawn]
    pairs = collections.Counter("%s -> %s" % (c.get("label_name"), c.get("pred_name")) for c in cbox)
    info = {"source": source, "images": len(rows), "seed_text": exp + "/unverified", "rule": UNVERIFIED_RULE,
            "near_dup_bits": NEAR_DUP_BITS,
            "verdicts": V._counter(collections.Counter(verdict[k] for k in drawn)),
            "conflict_boxes": len(cbox), "conflict_images": len({c.get("key") for c in cbox}),
            "conflict_pairs": dict(sorted(pairs.items())),
            "source_admit_summary": (step1["admit"].get("per_slug") or {}).get(source),
            "candidates": {s: {f: int(t[f]) for f in ("non_admitted", V.CONFLICT, V.UNKNOWN,
                                                       "excluded_near_dup", "eligible")}
                           for s, t in sorted(table.items())},
            "excluded_near_dup_examples": near[:20]}
    return rows, info


def relevance_file(relevance, step1_dir, testing, select_summary=None):
    """The relevance.json the verified draws are filtered by, or None: an
    explicit --relevance must exist; the default (relevance.json next to the
    base) is used when it exists. Without one, a production build refuses and
    a testing build warns. Given select's build summary, the file is checked
    here as select.increments will check it (relevance.load: made for this
    build, calibration check passed, statuses following from their numbers),
    so a refused file stops the build before anything is written."""
    from . import relevance as REL

    def checked(p):
        if select_summary is not None:
            try:
                REL.load(p, select_summary)
            except REL.RelevanceError as e:
                raise RealLoopError("relevance: %s" % e)
        return p
    if relevance is not None:
        p = Path(os.path.abspath(str(relevance)))
        if not p.is_file():
            raise RealLoopError("--relevance %s: no such file" % p)
        return checked(p)
    p = Path(step1_dir) / REL.OUT_NAME
    if p.is_file():
        return checked(p)
    msg = ("no %s: the verified and OTHER_HEAVY increments are drawn only from sources that pass the "
           "relevance filter (run inc.relevance build, or pass --relevance)" % p)
    if not testing:
        raise RealLoopError(msg)
    log("WARNING: %s; testing build: drawing them unfiltered" % msg)
    return None


def evidence_capacity(step1, n_verified, m, min_evidence):
    """--increment-sources evidence, before anything is written: the evidence
    as select.increments will load it (verify's admit_summary.json against the
    select build), applied to the increment pool, and the capacity of what
    stays (select.pool_capacity). Refuses unless it holds (N + 1) * M images
    (N verified increments and OTHER_HEAVY) and M of them in OtherPlant-heavy
    near-dup groups. Returns the capacity record."""
    V = S._verify()
    try:
        ev = S.load_evidence(step1["select"], V.ADMIT_SUMMARY, min_evidence)
        pool = sorted(C.read_manifest(step1["dir"] / S.POOL), key=lambda r: r["key"])
        cl = S.read_clusters(step1["dir"] / S.CLUSTERS)
        keep, info = S.apply_evidence(pool, ev, cl)
        cap = S.pool_capacity(keep, cl)
    except S.SelectError as e:
        raise RealLoopError(str(e))
    need = (int(n_verified) + 1) * int(m)
    held = {s: e["pool_images"] for s, e in info["evidenced_sources"].items() if e["pool_images"]}
    rec = {"min_evidence": ev["min_evidence"], "increment_images": int(m), "n_verified": int(n_verified),
           "needed_images": need, "needed_other_heavy_images": int(m), "evidenced_pool_images": cap["images"],
           "evidenced_other_heavy_images": cap["other_heavy_images"], "evidenced_sources_in_pool": held,
           "excluded_images": info["excluded_images"], "excluded_sources": len(info["excluded_sources"])}
    if cap["images"] < need or cap["other_heavy_images"] < m:
        bound = ("no OTHER_HEAVY increment of this size fits" if cap["other_heavy_images"] < m else
                 "the image count allows at most --n-verified %d" % (cap["images"] // m - 1))
        raise RealLoopError(
            "increment sources 'evidence': the evidenced increment pool cannot supply %d verified increments + "
            "OTHER_HEAVY of %d images. It holds %d images (%d needed), %d of them in OtherPlant-heavy near-dup "
            "groups (%d needed), from %d source(s) with at least %d verified cwd12-species box(es): %s; the %d "
            "increment-pool images of the %d other source(s) are excluded. At this M %s; lower --n-verified or "
            "--size, or build with --increment-sources relevance"
            % (n_verified, m, cap["images"], need, cap["other_heavy_images"], m, len(held), ev["min_evidence"],
               ", ".join("%s (%d)" % kv for kv in sorted(held.items())) or "none",
               info["excluded_images"], len(info["excluded_sources"]), bound))
    return rec


def same_base_baselines(base_sha):
    """Names of the 'baseline' experiments under INC_DIR whose base manifest
    has these bytes (base_b_v1 on the same base_B.jsonl): the loop's base arm
    repeats their cold runs, and its final runs read B's test again."""
    root = Path(C.INC_DIR)
    out = []
    for d in (sorted(root.iterdir()) if root.is_dir() else []):
        defn = S._read_json(d / "exp.json") if d.is_dir() else None
        if (isinstance(defn, dict) and defn.get("type") == "baseline"
                and (defn.get("base") or {}).get("manifest_sha256") == base_sha):
            out.append(d.name)
    return out


# ------------------------------------------------------ recovered increments
def _examples(items, n=3):
    return list(items)[:n]


def _abs_record_path(path):
    """A recorded path as an absolute Path; a relative one is relative to REPO
    (the funnel headers' convention for recorded paths)."""
    p = Path(str(path))
    if not p.is_absolute():
        p = Path(C.REPO) / p
    return Path(os.path.abspath(str(p)))


def _hash_record(rec, what):
    """(path, sha256) of a {"path", "sha256"} record whose file must still hash
    as recorded."""
    if not isinstance(rec, dict) or not rec.get("path") or not rec.get("sha256"):
        raise RealLoopError("%s: no {path, sha256} record" % what)
    p = _abs_record_path(rec["path"])
    if not p.is_file():
        raise RealLoopError("%s: %s does not exist" % (what, p))
    got = C.sha256_file(p)
    if got != rec["sha256"]:
        raise RealLoopError("%s: %s hashes to %s, recorded %s: it changed after recover wrote the overlay"
                            % (what, p, got[:12], str(rec["sha256"])[:12]))
    return p, got


def _read_rows(path, what):
    """JSON-lines rows of path, each an object."""
    rows = []
    with open(path) as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError as e:
                raise RealLoopError("%s line %d is not JSON: %s" % (what, i, e))
            if not isinstance(r, dict):
                raise RealLoopError("%s line %d is not an object" % (what, i))
            rows.append(r)
    return rows


def _as_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def overlay_exclusions(rows, dh, base_dh, dev_dh):
    """(eligible rows, record): the rows of the near_dup3 groups the draw may
    take (module docstring, "Near-duplicate exclusion"). dh maps a row's key to
    its unmasked dHash; base_dh and dev_dh map the base's pool images and the
    domain-dev images to theirs. A group is left out when a row lies within
    NEAR_DUP_BITS of a base or a domain-dev image, or when its rows lie in
    more than one pool."""
    indexes = []
    for reason, hashes in ((EXCLUDE_NEAR_BASE, base_dh), (EXCLUDE_NEAR_DOMAIN_DEV, dev_dh)):
        index = NearHashIndex()
        for k in sorted(hashes):
            index.add(int(hashes[k]), k, max_bits=NEAR_DUP_BITS)
        indexes.append((reason, index))
    reasons = collections.defaultdict(set)
    near = collections.defaultdict(list)
    pools_of = collections.defaultdict(set)
    for r in rows:
        g = str(r["near_dup3"])
        pools_of[g].add(str(r["pool"]))
        for reason, index in indexes:
            hit = index.find(int(dh[r["key"]]))
            if hit is not None:
                reasons[g].add(reason)
                near[g].append({"key": r["key"], "reason": reason, "near": hit[0], "bits": int(hit[1])})
    for g, p in pools_of.items():
        if len(p) > 1:
            reasons[g].add(EXCLUDE_CROSS_POOL)
    eligible = [r for r in rows if str(r["near_dup3"]) not in reasons]
    left = [r for r in rows if str(r["near_dup3"]) in reasons]
    by_reason = {}
    for reason in (EXCLUDE_NEAR_BASE, EXCLUDE_NEAR_DOMAIN_DEV, EXCLUDE_CROSS_POOL):
        gs = {g for g, rs in reasons.items() if reason in rs}
        rs_ = [r for r in left if str(r["near_dup3"]) in gs]
        by_reason[reason] = {"groups": len(gs), "images": len(rs_),
                             "by_pool": dict(sorted(collections.Counter(str(r["pool"]) for r in rs_).items()))}
    first = [{"near_dup3": g, "reasons": sorted(reasons[g]), "pools": sorted(pools_of[g]),
              "keys": sorted(r["key"] for r in left if str(r["near_dup3"]) == g)[:20], "near": near[g][:5]}
             for g in sorted(reasons)[:50]]
    record = {"bits": NEAR_DUP_BITS, "groups": len(reasons), "images": len(left),
              "by_pool": dict(sorted(collections.Counter(str(r["pool"]) for r in left).items())),
              "by_reason": by_reason, "first": first}
    return eligible, record


def load_overlay(step1_overlay, base_rows, guard, testing):
    """The recovery overlay of --step1-overlay, checked before anything is
    written (module docstring, "Recovered increments"). Returns {"dir",
    "recovery", "recovered_pool", "domain_dev", "rows", "eligible",
    "excluded", "dhash_unmasked", "inputs_checked", "quarantined_sources",
    "guard", "cross_pool_near_dup3"}; "eligible" are the rows the draw may
    take (overlay_exclusions)."""
    V = S._verify()
    d = Path(os.path.abspath(str(step1_overlay)))
    if not d.is_dir():
        raise RealLoopError("--step1-overlay %s: no such directory (run funnel recover)" % d)
    rec_path = d / RECOVERY_FILE
    rec = S._read_json(rec_path)
    if not isinstance(rec, dict):
        raise RealLoopError("%s is missing or not JSON (run funnel recover)" % rec_path)
    if rec.get("format") != RECOVERY_FORMAT:
        raise RealLoopError("%s has format %r, not %r" % (rec_path, rec.get("format"), RECOVERY_FORMAT))
    if rec.get("status") != RECOVERY_COMPLETE:
        raise RealLoopError("%s has status %r, not %r: recover did not finish (refusals: %s); nothing in the "
                            "overlay is an increment" % (rec_path, rec.get("status"), RECOVERY_COMPLETE,
                                                        _examples(rec.get("refusals") or [])))
    if rec.get("testing") and not testing:
        raise RealLoopError("%s was written by a testing recover; a production build refuses it" % rec_path)
    inputs = rec.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise RealLoopError("%s records no inputs, so its freshness cannot be checked" % rec_path)
    for name in sorted(inputs):
        _hash_record(inputs[name], "%s input %r" % (RECOVERY_FILE, name))
    guards = rec.get("guards") if isinstance(rec.get("guards"), dict) else {}
    nt = guards.get("never_train") if isinstance(guards.get("never_train"), dict) else {}
    h6 = guards.get("h6") if isinstance(guards.get("h6"), dict) else {}
    if nt.get("hits") != 0 or nt.get("unhashable") != 0 or h6.get("copies") != 0:
        raise RealLoopError("%s guards record never-train hits %r, unhashable images %r and H6 copies %r; each must "
                            "be 0" % (rec_path, nt.get("hits"), nt.get("unhashable"), h6.get("copies")))
    unchanged = rec.get("source_labels_unchanged") if isinstance(rec.get("source_labels_unchanged"), dict) else {}
    if unchanged.get("changed") != 0:
        raise RealLoopError("%s records %r changed source label(s) (source_labels_unchanged.changed must be 0)"
                            % (rec_path, unchanged.get("changed")))
    try:
        held, dd_rec = P.domain_dev_record(rec)
    except P.PilotError as e:
        raise RealLoopError(str(e))

    pool_path = d / RECOVERED_POOL_FILE
    if not pool_path.is_file():
        raise RealLoopError("no %s in %s" % (RECOVERED_POOL_FILE, d))
    pool_sha = C.sha256_file(pool_path)
    rp = rec.get("recovered_pool") if isinstance(rec.get("recovered_pool"), dict) else {}
    if rp.get("sha256") != pool_sha:
        raise RealLoopError("%s hashes to %s, %s records %s: the recovered pool is not the one recover wrote"
                            % (pool_path, pool_sha[:12], RECOVERY_FILE, str(rp.get("sha256"))[:12]))
    rows = _read_rows(pool_path, str(pool_path))
    if not rows:
        raise RealLoopError("%s lists no image" % pool_path)
    lacking = [i + 1 for i, r in enumerate(rows) if any(k not in r for k in OVERLAY_ROW_KEYS)]
    if lacking:
        raise RealLoopError("%s: %d row(s) lack some of %s, first line %d"
                            % (pool_path, len(lacking), list(OVERLAY_ROW_KEYS), lacking[0]))
    ungrouped = [r["key"] for r in rows if not isinstance(r["near_dup3"], str) or not r["near_dup3"]]
    if ungrouped:
        raise RealLoopError("%s: %d row(s) carry no near_dup3 group (a draw keeps groups whole, so a row without "
                            "one cannot be placed), e.g. %s" % (pool_path, len(ungrouped), ungrouped[:3]))
    keys = [r["key"] for r in rows]
    if len(set(keys)) != len(keys):
        dup = sorted(k for k, n in collections.Counter(keys).items() if n > 1)
        raise RealLoopError("%s lists %d key(s) twice, e.g. %s" % (pool_path, len(dup), dup[:3]))
    unknown = sorted({str(r["pool"]) for r in rows} - set(OVERLAY_POOLS))
    if unknown:
        raise RealLoopError("%s: pool(s) %s are not among %s" % (pool_path, unknown, list(OVERLAY_POOLS)))
    # the guard record must cover exactly these rows: every unmasked original and every masked copy
    n_masked = sum(1 for r in rows if r["image"] != r["unmasked_image"])
    coverage = {"never_train": (nt.get("unmasked_checked"), nt.get("masked_checked")),
                "h6": (h6.get("unmasked_checked"), h6.get("masked_checked"))}
    short = {k: v for k, v in sorted(coverage.items()) if v != (len(rows), n_masked)}
    if short:
        raise RealLoopError("%s guards do not cover %s: (unmasked, masked) checked %s, but the pool holds %d unmasked "
                            "originals and %d masked copies" % (rec_path, pool_path.name, short, len(rows), n_masked))
    # every drawable row was recovered under gates recovery.json records as passed
    gates = rec.get("gates") if isinstance(rec.get("gates"), dict) else {}
    ungated = [r["key"] for r in rows if r["pool"] in DRAWN_POOLS
               and (not isinstance(r["strata"], list) or not r["strata"]
                    or any(not isinstance(gates.get(s), dict) or gates[s].get("passed") is not True
                           for s in r["strata"]))]
    if ungated:
        raise RealLoopError("%d recovered row(s) name no stratum, or a stratum whose gate %s does not record as "
                            "passed, e.g. %s" % (len(ungated), RECOVERY_FILE, ungated[:3]))
    quarantined = sorted(rec.get("quarantined_sources") or [])
    q = [r["key"] for r in rows if r["source"] in set(quarantined)]
    if q:
        raise RealLoopError("%d recovered row(s) come from a source recover quarantined (%s), e.g. %s"
                            % (len(q), quarantined, q[:3]))
    h = [r["key"] for r in rows if r["key"] in held]
    if h:
        raise RealLoopError("%d recovered row(s) are held out as the H10d domain dev (%s), e.g. %s"
                            % (len(h), dd_rec["rows"]["path"], h[:3]))
    b_keys = {r["key"] for r in base_rows}
    b_images = {r["image"] for r in base_rows}
    b_shas = {r.get("sha256") for r in base_rows} - {None, ""}
    inbase = [r["key"] for r in rows if r["key"] in b_keys or r["image"] in b_images
              or r["unmasked_image"] in b_images or r["sha256"] in b_shas or r["unmasked_sha256"] in b_shas]
    if inbase:
        raise RealLoopError("%d recovered row(s) are images of the base (key, image path or image sha256), e.g. %s"
                            % (len(inbase), inbase[:3]))
    pool_rows = {r["key"]: r for r in C.read_manifest(V.POOL)}
    in_pool = [r for r in rows if r["pool"] != "FETCH"]
    missing = [r["key"] for r in in_pool if r["key"] not in pool_rows]
    if missing:
        raise RealLoopError("%d recovered row(s) of a pool other than FETCH are not verify's pool images (%s), "
                            "e.g. %s" % (len(missing), V.POOL, missing[:3]))
    wrong = [r["key"] for r in in_pool
             if (pool_rows[r["key"]]["image"], pool_rows[r["key"]]["sha256"], pool_rows[r["key"]]["source"])
             != (r["unmasked_image"], r["unmasked_sha256"], r["source"])]
    if wrong:
        raise RealLoopError("%d recovered row(s) name an unmasked original that is not their pool image (path, "
                            "sha256 or source differ from %s), e.g. %s" % (len(wrong), V.POOL, wrong[:3]))
    changed = [r["key"] for r in rows if not Path(r["unmasked_image"]).is_file()
               or C.sha256_file(r["unmasked_image"]) != r["unmasked_sha256"]]
    if changed:
        raise RealLoopError("%d recovered row(s) have an unmasked original that is missing or does not hash as "
                            "recorded, e.g. %s" % (len(changed), changed[:3]))

    # the never-train guard on the unmasked originals: pool_meta.jsonl's dHash
    try:
        dh = S.read_pool_dhash(V.POOL_META, [r["key"] for r in in_pool])
    except S.SelectError as e:
        raise RealLoopError(str(e))
    for r in rows:
        if r["pool"] == "FETCH":
            dh[r["key"]] = C.dhash(r["unmasked_image"])
    odd = [r["key"] for r in rows if dh.get(r["key"]) is None or _as_int(r["dhash_unmasked"]) != int(dh[r["key"]])]
    if odd:
        raise RealLoopError("%d recovered row(s) record a dhash_unmasked that is not their original's (%s for pool "
                            "images), e.g. %s" % (len(odd), V.POOL_META, odd[:3]))
    by_image = {r["unmasked_image"]: dh[r["key"]] for r in rows}
    hits, unhashable = guard.check([r["unmasked_image"] for r in rows], hash_fn=by_image.get)
    if hits or unhashable:
        raise RealLoopError("never-train guard on the unmasked originals of the recovered pool: %d image(s) within "
                            "%d dHash bits of an evaluation image, %d unhashable (fail closed); first: %s %s"
                            % (len(hits), C.HOLDOUT_NEAR_DUP_BITS, len(unhashable), hits[:3], unhashable[:3]))

    # near_dup3: rows within NEAR_DUP3_BITS of each other must share a group
    groups = S.dup_groups([int(dh[r["key"]]) for r in rows], bits=NEAR_DUP3_BITS)
    ids = collections.defaultdict(set)
    for g, r in zip(groups, rows):
        ids[int(g)].add(str(r["near_dup3"]))
    split = [sorted(v) for _g, v in sorted(ids.items()) if len(v) > 1]
    if split:
        raise RealLoopError("%d set(s) of recovered rows within %d dHash bits of each other carry different "
                            "near_dup3 groups, e.g. %s: the overlay's grouping does not match pool_meta.jsonl"
                            % (len(split), NEAR_DUP3_BITS, split[:2]))
    pools_of = collections.defaultdict(set)
    for r in rows:
        pools_of[str(r["near_dup3"])].add(str(r["pool"]))
    cross = {g: sorted(p) for g, p in sorted(pools_of.items()) if len(p) > 1}

    # near-duplicates of the base and of the domain dev (the base's train_core images are > 6 bits from every
    # pool image: verify's cwd12_copy stage)
    base_pool = sorted(r["key"] for r in base_rows if r["key"] in pool_rows)
    not_pool = sorted(k for k in held if k not in pool_rows)
    if not_pool:
        raise RealLoopError("the H10d domain dev %s holds %d key(s) that are not verify's pool images, so their "
                            "near-duplicates cannot be found, e.g. %s" % (dd_rec["rows"]["path"], len(not_pool),
                                                                          not_pool[:3]))
    try:
        base_dh = S.read_pool_dhash(V.POOL_META, base_pool) if base_pool else {}
        dev_dh = S.read_pool_dhash(V.POOL_META, sorted(held)) if held else {}
    except S.SelectError as e:
        raise RealLoopError(str(e))
    eligible, excluded = overlay_exclusions(rows, dh, base_dh, dev_dh)

    # the images as they are trained (masked copies): inc/train.py's check and the never-train guard
    try:
        _rows, _dh, info = P.check_training_manifest(pool_path, guard=guard, what="the recovered pool")
    except P.PilotError as e:
        raise RealLoopError(str(e))
    return {"dir": d, "rows": rows, "eligible": eligible, "excluded": excluded, "dhash_unmasked": dh,
            "recovery": {"path": str(rec_path), "sha256": C.sha256_file(rec_path), "status": rec.get("status"),
                         "testing": bool(rec.get("testing"))},
            "recovered_pool": {"path": str(pool_path), "sha256": pool_sha, "rows": len(rows)},
            "domain_dev": dd_rec,
            "inputs_checked": len(inputs), "quarantined_sources": quarantined,
            "guard": {"unmasked": {"checked": len(rows), "hits": 0, "unhashable": 0,
                                   "dhash_from": {"pool_meta": len(in_pool), "file": len(rows) - len(in_pool)}},
                      "trained": info["guard"]},
            "cross_pool_near_dup3": cross}


def pool_groups(rows):
    """{pool: [group]} for the pools the sequence draws from: each pool's rows
    grouped by near_dup3, each group's rows sorted by key, the groups sorted
    by their first key."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[str(r["pool"])][str(r["near_dup3"])].append(r)
    out = {}
    for pool in sorted(set(RECOVERED_POOLS.values())):
        groups = [sorted(g, key=lambda r: r["key"]) for g in by.get(pool, {}).values()]
        out[pool] = sorted(groups, key=lambda g: g[0]["key"])
    return out


def plan_recovered(pools, m, base_n):
    """Which pool each step of RECOVERED_SEQUENCE is drawn from, and M_eff
    (module docstring, "Plan"). pools = {pool: [group]}. Returns {"m",
    "m_requested", "floor", "base_images", "capacity", "steps", "dropped",
    "substituted"}; refuses when M_eff is below the floor or a dropped arm's
    step has no substitute."""
    m = int(m)
    if m < 1:
        raise RealLoopError("--size must be >= 1, got %r" % m)
    floor = RECOVERED_FLOOR_FRAC * int(base_n)
    steps_of = collections.Counter(RECOVERED_POOLS[s] for s in RECOVERED_SEQUENCE)
    capacity = {}
    for p in sorted(steps_of):
        n = sum(len(g) for g in pools.get(p, []))
        capacity[p] = {"images": n, "groups": len(pools.get(p, [])), "steps": steps_of[p],
                       "per_step": n / float(steps_of[p])}
    kept = [p for p in sorted(steps_of) if capacity[p]["per_step"] >= floor]
    dropped = [{"pool": p, "images": capacity[p]["images"], "steps": capacity[p]["steps"],
                "per_step": capacity[p]["per_step"], "floor": floor,
                "reason": "capacity per step %.2f below the floor %.2f (%.2f x %d)"
                          % (capacity[p]["per_step"], floor, RECOVERED_FLOOR_FRAC, base_n)}
               for p in sorted(steps_of) if p not in kept]
    table = ", ".join("%s %d images / %d steps = %.2f" % (p, c["images"], c["steps"], c["per_step"])
                      for p, c in sorted(capacity.items()))
    if not kept:
        raise RealLoopError("no recovered arm reaches the floor of M, %.2f images per step (%.2f x |B| = %d): %s"
                            % (floor, RECOVERED_FLOOR_FRAC, base_n, table))
    m_eff = min(m, int(math.floor(min(capacity[p]["per_step"] for p in kept))))
    if m_eff < floor:
        raise RealLoopError("M would be %d, below the floor %.2f (%.2f x |B| = %d); requested %d; %s"
                            % (m_eff, floor, RECOVERED_FLOOR_FRAC, base_n, m, table))
    steps, used, substituted = {}, collections.Counter(), []
    for s in RECOVERED_SEQUENCE:
        p = RECOVERED_POOLS[s]
        if p in kept:
            steps[s] = p
            used[p] += 1
    for s in RECOVERED_SEQUENCE:
        p = RECOVERED_POOLS[s]
        if p in kept:
            continue
        remaining = {q: capacity[q]["images"] - used[q] * m_eff for q in SUBSTITUTION_ORDER if q in kept}
        cands = [q for q in SUBSTITUTION_ORDER if q in remaining and remaining[q] >= m_eff]
        if not cands:
            raise RealLoopError("step %s (arm %s, dropped) has no substitute: no arm among %s has %d images left "
                                "(remaining %s)" % (s, p, list(SUBSTITUTION_ORDER), m_eff, remaining))
        best = max(cands, key=lambda q: (remaining[q], -SUBSTITUTION_ORDER.index(q)))
        steps[s] = best
        used[best] += 1
        substituted.append({"step": s, "from": p, "to": best, "remaining_before": remaining[best]})
    return {"m": m_eff, "m_requested": m, "floor": floor, "base_images": int(base_n), "capacity": capacity,
            "steps": {s: steps[s] for s in RECOVERED_SEQUENCE}, "dropped": dropped, "substituted": substituted}


def draw_recovered(exp, step, pool_groups, m):
    """The step's rows: whole groups of pool_groups (the pool's groups no
    earlier step took), sorted by first key, in the order of
    numpy.random.default_rng(stable_int(exp + "/" + step)).permutation; a group
    is added while it fits m exactly, skipped when it would overshoot; refuses
    if m cannot be reached."""
    import numpy as np
    groups = sorted(pool_groups, key=lambda g: g[0]["key"])
    order = np.random.default_rng(C.stable_int(exp + "/" + step)).permutation(len(groups))
    out, n = [], 0
    for i in order:
        g = groups[int(i)]
        if n + len(g) <= m:
            out.extend(g)
            n += len(g)
            if n == m:
                break
    if n != m:
        raise RealLoopError("step %s: whole near-dup groups reach %d images, not %d (the pool left %d images in %d "
                            "groups)" % (step, n, m, sum(len(g) for g in groups), len(groups)))
    return sorted(out, key=lambda r: r["key"])


def _build_recovered(exp, base, size, replay_mode, recipes, truth, testing, backend, guard, init, quiet,
                     gate_flips_mode, step1_overlay):
    """build() with --increment-sources recovered (module docstring,
    "Recovered increments")."""
    gate = P.gate_block(gate_flips_mode)
    names = parse_recipes(recipes)
    seq = list(RECOVERED_SEQUENCE)
    testing = P._check_testing(testing)
    paths = D.Paths(exp)
    P._check_new(paths)
    base = Path(os.path.abspath(str(base or (S.step1_dir() / S.BASE_B))))
    nt = P.never_train_status(testing=bool(testing))
    if guard is None:
        try:
            guard = S._load_guard()
        except S.SelectError as e:
            raise RealLoopError(str(e))
    step1 = load_step1(base)
    provenance = P.select_provenance(step1["select"], testing=bool(testing), what="base %s" % base)
    same_base = same_base_baselines(step1["base_sha256"])
    base_pre = C.read_manifest(base)
    overlay = load_overlay(step1_overlay, base_pre, guard, bool(testing))
    pools = pool_groups(overlay["eligible"])
    plan = plan_recovered(pools, size, len(base_pre))
    m = plan["m"]
    remaining = {p: list(gs) for p, gs in pools.items()}
    drawn = {}
    for s in seq:
        p = plan["steps"][s]
        rows_s = draw_recovered(exp, s, remaining[p], m)
        taken = {r["key"] for r in rows_s}
        remaining[p] = [g for g in remaining[p] if g[0]["key"] not in taken]
        drawn[s] = rows_s
    if same_base:
        log("NOTE: %s already train this base (same bytes); this loop's base arm repeats their cold runs "
            "and its final runs read base B's test again" % ", ".join(same_base))

    # the base, copied (same bytes) and checked as its runs will check it
    base_copy = paths.manifests / ("%s.jsonl" % BASE_NAME)
    base_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base, base_copy)
    if C.sha256_file(base_copy) != step1["base_sha256"]:
        raise RealLoopError("copy of %s does not hash like the original" % base)
    base_rows, _base_dh, base_info = P.check_training_manifest(base_copy, guard=guard, what="the base")
    if len(base_rows) != len(base_pre):
        raise RealLoopError("internal: the base holds %d rows, the plan used %d" % (len(base_rows), len(base_pre)))

    files = {s: paths.manifests / ("%s.jsonl" % s) for s in seq}
    for s in seq:
        C.write_manifest(files[s], [{k: r[k] for k in C.MANIFEST_KEYS} for r in drawn[s]])
    checked = {}
    for s in seq:
        r, _dh, info = P.check_training_manifest(files[s], guard=guard, what="increment %s" % s)
        if len(r) != m:
            raise RealLoopError("increment %s holds %d images, not %d" % (s, len(r), m))
        checked[s] = (r, info)
    parts = [(BASE_NAME, base_rows)] + [(s, checked[s][0]) for s in seq]
    for i, (a, ra) in enumerate(parts):
        for b, rb in parts[:i]:
            D.check_disjoint(rb, ra, a, "%s" % b)

    subst = {x["step"]: x["from"] for x in plan["substituted"]}
    shas = {s: C.sha256_file(files[s]) for s in seq}
    per_step = {}
    for s in seq:
        per_step[s] = {"pool": plan["steps"][s], "substituted_from": subst.get(s),
                       "images": len(drawn[s]), "groups": len({str(r["near_dup3"]) for r in drawn[s]}),
                       "policy_counts": dict(sorted(collections.Counter(str(r["policy"]) for r in drawn[s]).items())),
                       "sources": dict(sorted(collections.Counter(r["source"] for r in drawn[s]).items())),
                       "boxes": C.class_counts(checked[s][0])}
    table = P.inc_recipes()
    recipe_defs = {r: table[r] for r in names}
    cold = P.cold_recipe()
    warmup = P.warmup_table(recipe_defs, cold, len(base_rows), [(s, m, False) for s in seq], replay_mode)
    sel_in = step1["select"].get("inputs") or {}
    overlay_rec = {"dir": str(overlay["dir"]), "recovery_sha256": overlay["recovery"]["sha256"],
                   "recovered_pool_sha256": overlay["recovered_pool"]["sha256"]}
    step1_files = dict(step1["files"], select_summary=step1["select_summary"],
                       train_core={k: (sel_in.get("train_core") or {}).get(k) for k in ("path", "sha256")},
                       never_train={k: (sel_in.get("never_train") or {}).get(k) for k in ("path", "sha256")},
                       increment_sources={"mode": SOURCES_RECOVERED, "overlay": overlay_rec, "rule": RECOVERED_RULE,
                                          "m_requested": plan["m_requested"], "m": m, "dropped": plan["dropped"],
                                          "substituted": plan["substituted"]})
    defn = {
        "exp": exp, "type": "chain", "builder": BUILDER, "testing": testing, "replay_mode": replay_mode,
        "gate": gate, "seeds": list(D.SEEDS), "init_weights": D.COLD_INIT, "decision_exam": D.DECISION_EXAM,
        "final_exams": list(D.FINAL_EXAMS),
        "base": P._entry(BASE_NAME, base_copy, base_rows, step1["base_sha256"], recipe=cold, source_manifest=str(base)),
        "steps": [P._entry(s, files[s], drawn[s], shas[s], clean=False, kind=KIND_RECOVERED, pool=plan["steps"][s],
                           policy_counts=per_step[s]["policy_counts"], substituted_from=subst.get(s))
                  for s in seq],
        "recipes": recipe_defs, "truth": True, "truth_recipe": cold,
        "increment_images": m, "step1": step1_files,
        "effective_warmup": warmup, "attribution_scope": RECOVERED_ATTRIBUTION_SCOPE,
    }
    pool_info = {}
    for p in OVERLAY_POOLS:
        rs = [r for r in overlay["rows"] if r["pool"] == p]
        if rs:
            ok = [r for r in overlay["eligible"] if r["pool"] == p]
            pool_info[p] = {"images": len(rs), "groups": len({str(r["near_dup3"]) for r in rs}),
                            "eligible_images": len(ok), "eligible_groups": len({str(r["near_dup3"]) for r in ok}),
                            "drawn": p in set(RECOVERED_POOLS.values()),
                            "policy_counts": dict(sorted(collections.Counter(str(r["policy"]) for r in rs).items())),
                            "sources": dict(sorted(collections.Counter(r["source"] for r in rs).items()))}
    cross = overlay["cross_pool_near_dup3"]
    summary = {
        "exp": exp, "testing": bool(testing), "built_utc": D._utc(), "builder": BUILDER,
        "replay_mode": replay_mode, "gate": dict(gate), "truth": True, "recipes": names,
        "size": {"images_per_increment": m, "requested": plan["m_requested"], "floor": plan["floor"],
                 "floor_frac": RECOVERED_FLOOR_FRAC, "base_images": len(base_rows)},
        "never_train": nt, "step1": step1_files,
        "select_provenance": provenance, "same_base_baselines": same_base,
        "base": dict(P.manifest_summary(base_rows, step1["base_sha256"], base_info), name=BASE_NAME, source=str(base),
                     copy=str(base_copy)),
        "increments": {s: dict(P.manifest_summary(checked[s][0], shas[s], checked[s][1]), kind=KIND_RECOVERED,
                               clean=False, position=i, manifest=str(files[s]), pool=plan["steps"][s],
                               substituted_from=subst.get(s), policy_counts=per_step[s]["policy_counts"])
                       for i, s in enumerate(seq, 1)},
        "recovered": {"rule": RECOVERED_RULE,
                      "overlay": dict(overlay_rec, recovery=overlay["recovery"],
                                      recovered_pool=overlay["recovered_pool"], domain_dev=overlay["domain_dev"],
                                      inputs_checked=overlay["inputs_checked"],
                                      quarantined_sources=overlay["quarantined_sources"]),
                      "pools": pool_info, "plan": plan, "per_step": per_step, "guard": overlay["guard"],
                      "excluded": overlay["excluded"],
                      "cross_pool_near_dup3": {"groups": len(cross), "first": dict(list(cross.items())[:50])}},
        "sequence": seq, "clean": [],
        "recipe_definitions": recipe_defs, "cold_recipe": cold, "warmup": warmup,
        "attribution_scope": RECOVERED_ATTRIBUTION_SCOPE,
    }
    D._write_json(paths.root / P.BUILD_SUMMARY, summary)
    log("%s: base %d images; %d recovered increments of %d (requested %d, floor %.2f): %s; dropped %s; "
        "substituted %s; near-duplicate groups left out %d (%d images: %s); recipes %s; replay %s; gate flips %s; "
        "truth arm on"
        % (exp, len(base_rows), len(seq), m, plan["m_requested"], plan["floor"],
           ", ".join("%s<-%s" % (s, plan["steps"][s]) for s in seq), [x["pool"] for x in plan["dropped"]] or "none",
           ["%s<-%s" % (x["step"], x["to"]) for x in plan["substituted"]] or "none",
           overlay["excluded"]["groups"], overlay["excluded"]["images"],
           {k: v["groups"] for k, v in overlay["excluded"]["by_reason"].items()}, names, replay_mode,
           gate["flips_mode"]))
    result = D.Driver(exp, backend=backend, quiet=quiet).init(defn) if init else None
    return summary, defn, result


# ----------------------------------------------------------------- build
def build(exp, base=None, n_verified=None, size=None, replay_mode=None, recipes=None, truth=True,
          testing=False, backend=None, guard=None, init=True, quiet=False, workers=S.WORKERS, relevance=None,
          gate_flips_mode=D.DEFAULT_FLIPS_MODE, increment_sources=S.SOURCES_RELEVANCE, min_evidence=None,
          step1_overlay=None):
    """Build the Steps 2-3 chain experiment, then driver init. Returns
    (summary, definition, init result or None). n_verified None means
    N_VERIFIED ('relevance' and 'evidence'); 'recovered' refuses any value."""
    if replay_mode not in D.REPLAY_MODES:
        raise RealLoopError("replay mode %r not in %s" % (replay_mode, D.REPLAY_MODES))
    if gate_flips_mode not in D.FLIPS_MODES:
        raise RealLoopError("gate flips mode %r not in %s" % (gate_flips_mode, D.FLIPS_MODES))
    if increment_sources not in INCREMENT_SOURCE_MODES:
        raise RealLoopError("increment sources %r not in %s" % (increment_sources, INCREMENT_SOURCE_MODES))
    if increment_sources == SOURCES_RECOVERED:
        given = [flag for flag, v in (("--relevance", relevance), ("--min-evidence", min_evidence),
                                      ("--n-verified", n_verified)) if v is not None]
        if given:
            raise RealLoopError("--increment-sources recovered draws its steps from the recovery overlay; it refuses "
                                "%s (the sequence is fixed: %s)" % (", ".join(given), ", ".join(RECOVERED_SEQUENCE)))
        if step1_overlay is None:
            raise RealLoopError("--increment-sources recovered needs --step1-overlay (INC_DIR/step1_r1)")
        if size is None:
            raise RealLoopError("--increment-sources recovered needs --size (M, the requested images per step)")
        if not truth:
            raise RealLoopError("--increment-sources recovered keeps the truth arm on: it is the loop's measurement "
                                "of the recovered data (--no-truth refused)")
        return _build_recovered(exp, base, size, replay_mode, recipes, truth, testing, backend, guard, init, quiet,
                                gate_flips_mode, step1_overlay)
    if step1_overlay is not None:
        raise RealLoopError("--step1-overlay applies to --increment-sources recovered only")
    n_verified = N_VERIFIED if n_verified is None else n_verified
    evidence = increment_sources == S.SOURCES_EVIDENCE
    if evidence and relevance is not None:
        raise RealLoopError("--increment-sources evidence does not read a relevance file (--relevance): the "
                            "source-evidence criterion replaces the relevance filter")
    if not evidence and min_evidence is not None:
        raise RealLoopError("--min-evidence applies to --increment-sources evidence only")
    if evidence:
        min_evidence = S.MIN_EVIDENCE if min_evidence is None else min_evidence
        if isinstance(min_evidence, bool) or not isinstance(min_evidence, int) or min_evidence < 1:
            raise RealLoopError("--min-evidence must be an integer >= 1, got %r" % (min_evidence,))
    gate = P.gate_block(gate_flips_mode)
    names = parse_recipes(recipes)
    seq = sequence(n_verified)
    n_verified = int(n_verified)
    if len(seq) < MIN_DECIDED:
        log("WARNING: %d increments; the protocol's acceptance needs at least %d decided"
            % (len(seq), MIN_DECIDED))
    if size is not None and int(size) < 1:
        raise RealLoopError("--size must be >= 1, got %r" % size)
    testing = P._check_testing(testing)
    paths = D.Paths(exp)
    P._check_new(paths)
    base = Path(os.path.abspath(str(base or (S.step1_dir() / S.BASE_B))))
    nt = P.never_train_status(testing=bool(testing))
    if guard is None:
        try:
            guard = S._load_guard()
        except S.SelectError as e:
            raise RealLoopError(str(e))
    step1 = load_step1(base)
    provenance = P.select_provenance(step1["select"], testing=bool(testing), what="base %s" % base)
    if evidence:
        # no relevance file; the evidence and the evidenced pool's capacity, before anything is written
        rel_path = None
        m_pre = int(size) if size is not None else default_size(len(C.read_manifest(base)))
        capacity = evidence_capacity(step1, n_verified, m_pre, min_evidence)
    else:
        rel_path = relevance_file(relevance, step1["dir"], bool(testing), step1["select"])
        capacity = None
    same_base = same_base_baselines(step1["base_sha256"])
    if same_base:
        log("NOTE: %s already train this base (same bytes); this loop's base arm repeats their cold runs "
            "and its final runs read base B's test again" % ", ".join(same_base))
    V = S._verify()

    # the base, copied (same bytes) and checked as its runs will check it
    base_copy = paths.manifests / ("%s.jsonl" % BASE_NAME)
    base_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(base, base_copy)
    if C.sha256_file(base_copy) != step1["base_sha256"]:
        raise RealLoopError("copy of %s does not hash like the original" % base)
    base_rows, base_dh, base_info = P.check_training_manifest(base_copy, guard=guard, what="the base")
    m = int(size) if size is not None else default_size(len(base_rows))
    if evidence and m != m_pre:
        raise RealLoopError("internal: M is %d, the capacity check used %d" % (m, m_pre))

    # V1..VN and OTHER_HEAVY: select's own draw, into this experiment's manifests/
    ev_kw = ({"sources": S.SOURCES_EVIDENCE, "min_evidence": min_evidence, "admit_summary": V.ADMIT_SUMMARY}
             if evidence else {})
    try:
        inc_summary = S.increments(exp, n_verified, m, other_heavy=True, base_dir=step1["dir"],
                                   exp_dir=paths.root, workers=workers, relevance=rel_path, **ev_kw)
    except S.SelectError as e:
        raise RealLoopError("select increments: %s" % e)
    ev_info = inc_summary.get("evidence") if evidence else None
    if evidence and (ev_info or {}).get("admit_summary", {}).get("sha256") != step1["files"]["admit_summary"]["sha256"]:
        raise RealLoopError("the admit_summary.json select's draw read is not the one Step 1 was checked with "
                            "(it changed during the build)")
    files = {verified_name(j): paths.manifests / (S.INC_NAME % j) for j in range(1, n_verified + 1)}
    files[OTHER_HEAVY] = paths.manifests / S.INC_OTHER
    rows = {n: C.read_manifest(p) for n, p in files.items()}
    try:
        inc_dh = S.read_pool_dhash(V.POOL_META, [r["key"] for rs in rows.values() for r in rs])
    except S.SelectError as e:
        raise RealLoopError(str(e))

    # UNVERIFIED
    taken = [(BASE_NAME, base_rows, {r["key"]: base_dh[r["image"]] for r in base_rows})]
    taken += [(n, rows[n], inc_dh) for n in seq if n in rows]
    un_rows, un_info = draw_unverified(exp, m, step1, taken, guard)
    rel_info = inc_summary.get("relevance") if rel_path is not None else None
    if rel_info is not None:
        un_info["source_relevance"] = rel_info["source_status"].get(
            un_info["source"], "no image in the increment pool")
    if ev_info is not None:              # recorded only: UNVERIFIED is never filtered
        n_ver = ((((step1["admit"].get("per_slug") or {}).get(un_info["source"]) or {}).get("boxes") or {})
                 .get(V.VERIFIED, 0))
        un_info["source_evidence"] = {"verified_boxes": n_ver, "evidenced": n_ver >= ev_info["min_evidence"]}
    files[UNVERIFIED] = paths.manifests / UNVERIFIED_MANIFEST
    C.write_manifest(files[UNVERIFIED], un_rows)
    rows[UNVERIFIED] = un_rows

    # every increment as its runs will check it; sizes; pairwise disjoint
    checked = {}
    for n in seq:
        r, _dh, info = P.check_training_manifest(files[n], guard=guard, what="increment %s" % n)
        if len(r) != m:
            raise RealLoopError("increment %s holds %d images, not %d" % (n, len(r), m))
        checked[n] = (r, info)
    parts = [(BASE_NAME, base_rows)] + [(n, checked[n][0]) for n in seq]
    for i, (a, ra) in enumerate(parts):
        for b, rb in parts[:i]:
            D.check_disjoint(rb, ra, a, "%s" % b)

    clean = [n for n in seq if n != UNVERIFIED]
    kind = {n: KIND_VERIFIED for n in seq}
    kind.update({OTHER_HEAVY: KIND_OTHER_HEAVY, UNVERIFIED: KIND_UNVERIFIED})
    shas = {n: C.sha256_file(files[n]) for n in seq}
    extra = {n: {"kind": kind[n], "select_manifest": files[n].name} for n in seq}
    extra[UNVERIFIED] = {"kind": KIND_UNVERIFIED, "source": un_info["source"],
                         "verdicts": un_info["verdicts"], "rule": UNVERIFIED_RULE}
    table = P.inc_recipes()
    recipe_defs = {r: table[r] for r in names}
    cold = P.cold_recipe()
    warmup = P.warmup_table(recipe_defs, cold, len(base_rows), [(n, m, n in clean) for n in seq], replay_mode)
    base_sha = step1["base_sha256"]
    sel_in = step1["select"].get("inputs") or {}
    step1_files = dict(step1["files"], select_summary=step1["select_summary"],
                       increments_summary={"path": str(paths.manifests / S.INC_SUMMARY),
                                           "sha256": C.sha256_file(paths.manifests / S.INC_SUMMARY)},
                       train_core={k: (sel_in.get("train_core") or {}).get(k) for k in ("path", "sha256")},
                       never_train={k: (sel_in.get("never_train") or {}).get(k) for k in ("path", "sha256")},
                       relevance=({"path": rel_info["path"], "sha256": rel_info["sha256"]}
                                  if rel_info is not None else None))
    if evidence:     # only here: a default build's definition is the one built before --increment-sources existed
        step1_files["increment_sources"] = {"mode": S.SOURCES_EVIDENCE, "min_evidence": ev_info["min_evidence"],
                                            "rule": ev_info["rule"]}
    defn = {
        "exp": exp, "type": "chain", "builder": BUILDER, "testing": testing, "replay_mode": replay_mode,
        "gate": gate, "seeds": list(D.SEEDS), "init_weights": D.COLD_INIT, "decision_exam": D.DECISION_EXAM,
        "final_exams": list(D.FINAL_EXAMS),
        "base": P._entry(BASE_NAME, base_copy, base_rows, base_sha, recipe=cold, source_manifest=str(base)),
        "steps": [P._entry(n, files[n], rows[n], shas[n], clean=n in clean, **extra[n]) for n in seq],
        "recipes": recipe_defs, "truth": bool(truth), "truth_recipe": cold,
        "increment_images": m, "step1": step1_files,
        "effective_warmup": warmup, "attribution_scope": ATTRIBUTION_SCOPE,
    }
    summary = {
        "exp": exp, "testing": bool(testing), "built_utc": D._utc(), "builder": BUILDER,
        "replay_mode": replay_mode, "gate": dict(gate), "truth": bool(truth), "recipes": names,
        "size": {"images_per_increment": m, "default": size is None, "inc_frac": S.INC_FRAC,
                 "base_images": len(base_rows)},
        "n_verified": n_verified, "never_train": nt, "step1": step1_files,
        "select_provenance": provenance, "same_base_baselines": same_base,
        "base": dict(P.manifest_summary(base_rows, base_sha, base_info), name=BASE_NAME, source=str(base),
                     copy=str(base_copy)),
        "increments": {n: dict(P.manifest_summary(checked[n][0], shas[n], checked[n][1]), kind=kind[n],
                               clean=n in clean, position=i, manifest=str(files[n]))
                       for i, n in enumerate(seq, 1)},
        "select_increments": {"seed_text": exp, "other_heavy_seed_text": exp + "/otherplant",
                              "params": inc_summary.get("params"), "per_increment": inc_summary.get("increments")},
        "unverified": un_info,
        "relevance": (dict((k, v) for k, v in rel_info.items() if k != "source_status")
                      if rel_info is not None else
                      {"applied": False, "reason": "increment sources 'evidence': the source-evidence criterion "
                                                   "('evidence') replaces the relevance filter"}
                      if evidence else
                      {"applied": False, "reason": "testing build without a relevance file"}),
        "sequence": seq, "clean": clean,
        "recipe_definitions": recipe_defs, "cold_recipe": cold, "warmup": warmup,
        "attribution_scope": ATTRIBUTION_SCOPE,
    }
    if evidence:
        summary["evidence"] = dict(ev_info, capacity=capacity)
    D._write_json(paths.root / P.BUILD_SUMMARY, summary)
    log("%s: base %d images; %d increments of %d (%s; sources: %s); UNVERIFIED from %s (%s); recipes %s; "
        "replay %s; gate flips %s; truth arm %s"
        % (exp, len(base_rows), len(seq), m, ", ".join(seq), increment_sources, un_info["source"],
           un_info["verdicts"], names, replay_mode, gate["flips_mode"], "on" if truth else "off"))
    result = D.Driver(exp, backend=backend, quiet=quiet).init(defn) if init else None
    return summary, defn, result


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m weed_optimizer_framework.tools.inc.realloop",
                                 description="INC Steps 2-3: build the incremental loop on real data.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the chain experiment and call driver init")
    b.add_argument("--exp", required=True)
    b.add_argument("--base", default=None, help="base manifest (default INC_DIR/step1/base_B.jsonl)")
    b.add_argument("--n-verified", type=int, default=None,
                   help="verified increments (default %d), plus OTHER_HEAVY and UNVERIFIED; refused with "
                        "--increment-sources recovered" % N_VERIFIED)
    b.add_argument("--size", type=int, default=None,
                   help="images per increment (default %d%% of the base's images; required with --increment-sources "
                        "recovered, where it is the requested M)" % round(100 * S.INC_FRAC))
    b.add_argument("--replay-mode", choices=D.REPLAY_MODES, required=True,
                   help="'sample' (cand on D_k + R1, null on R1 + R2) or 'full' (cand on the accepted "
                        "pool + D_k, null on the pool)")
    b.add_argument("--recipes", required=True, help="comma-separated, among %s" % sorted(P.inc_recipes()))
    b.add_argument("--gate-flips-mode", choices=D.FLIPS_MODES, default=D.DEFAULT_FLIPS_MODE,
                   help="what the gate's flips guard counts: 'negative' (default; protocol v1) or 'net' "
                        "(protocol v2: negative - positive flips)")
    b.add_argument("--no-truth", action="store_true", help="no truth arm")
    b.add_argument("--increment-sources", choices=INCREMENT_SOURCE_MODES, default=S.SOURCES_RELEVANCE,
                   help="the relevance criterion of V* and OTHER_HEAVY: 'relevance' (default: the relevance file "
                        "below) or 'evidence' (only sources with --min-evidence verified cwd12-species boxes; no "
                        "relevance file); or 'recovered': the recovered sequence %s from --step1-overlay "
                        "(realloop_v2)" % ", ".join(RECOVERED_SEQUENCE))
    b.add_argument("--step1-overlay", default=None,
                   help="--increment-sources recovered: the recovery overlay directory (INC_DIR/step1_r1) with "
                        "recovery.json and recovered_pool.jsonl")
    b.add_argument("--relevance", default=None,
                   help="inc.relevance's relevance.json: V* and OTHER_HEAVY come only from sources that pass it "
                        "(default: relevance.json next to --base; a production build refuses without one)")
    b.add_argument("--min-evidence", type=int, default=None,
                   help="--increment-sources evidence: verified cwd12-species boxes a source needs (default %d)"
                        % S.MIN_EVIDENCE)
    b.add_argument("--testing", action="store_true",
                   help="gate on test-mode scores (needs %s=1); every output says TESTING" % TEST_ENV)
    b.add_argument("--testing-settings", default=None,
                   help="as --testing, with inc/train.py's test-mode settings as a JSON object")
    b.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        build(a.exp, base=a.base, n_verified=a.n_verified, size=a.size, replay_mode=a.replay_mode,
              recipes=a.recipes, truth=not a.no_truth, testing=P.testing_arg(a.testing, a.testing_settings),
              quiet=a.quiet, relevance=a.relevance, gate_flips_mode=a.gate_flips_mode,
              increment_sources=a.increment_sources, min_evidence=a.min_evidence, step1_overlay=a.step1_overlay)
    except (P.PilotError, D.DriverError, S.SelectError) as e:
        print("[inc.realloop] ERROR: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
