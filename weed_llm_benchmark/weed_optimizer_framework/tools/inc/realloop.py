"""INC Steps 2-3, the incremental loop on real data: the experiment builder
(docs/INCREMENTAL_PROTOCOL.md, "Steps 2-3"; docs/INCREMENTAL_PROTOCOL_RUNNER.md,
"Real-loop build").

    python -m weed_optimizer_framework.tools.inc.realloop build --exp real_v1
        --replay-mode {sample,full} --recipes full[,freeze,lora] [--gate-flips-mode {negative,net}]
        [--base INC_DIR/step1/base_B.jsonl] [--n-verified 6] [--size M] [--no-truth]
        [--increment-sources relevance] [--relevance INC_DIR/step1/relevance.json]
        | --increment-sources evidence [--min-evidence 1]
        [--testing | --testing-settings JSON] [--quiet]

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
"""
from __future__ import annotations

import argparse
import collections
import csv
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


# ----------------------------------------------------------------- build
def build(exp, base=None, n_verified=N_VERIFIED, size=None, replay_mode=None, recipes=None, truth=True,
          testing=False, backend=None, guard=None, init=True, quiet=False, workers=S.WORKERS, relevance=None,
          gate_flips_mode=D.DEFAULT_FLIPS_MODE, increment_sources=S.SOURCES_RELEVANCE, min_evidence=None):
    """Build the Steps 2-3 chain experiment, then driver init. Returns
    (summary, definition, init result or None)."""
    if replay_mode not in D.REPLAY_MODES:
        raise RealLoopError("replay mode %r not in %s" % (replay_mode, D.REPLAY_MODES))
    if gate_flips_mode not in D.FLIPS_MODES:
        raise RealLoopError("gate flips mode %r not in %s" % (gate_flips_mode, D.FLIPS_MODES))
    if increment_sources not in S.SOURCE_MODES:
        raise RealLoopError("increment sources %r not in %s" % (increment_sources, S.SOURCE_MODES))
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
    b.add_argument("--n-verified", type=int, default=N_VERIFIED,
                   help="verified increments (default %(default)s), plus OTHER_HEAVY and UNVERIFIED")
    b.add_argument("--size", type=int, default=None,
                   help="images per increment (default %d%% of the base's images)" % round(100 * S.INC_FRAC))
    b.add_argument("--replay-mode", choices=D.REPLAY_MODES, required=True,
                   help="'sample' (cand on D_k + R1, null on R1 + R2) or 'full' (cand on the accepted "
                        "pool + D_k, null on the pool)")
    b.add_argument("--recipes", required=True, help="comma-separated, among %s" % sorted(P.inc_recipes()))
    b.add_argument("--gate-flips-mode", choices=D.FLIPS_MODES, default=D.DEFAULT_FLIPS_MODE,
                   help="what the gate's flips guard counts: 'negative' (default; protocol v1) or 'net' "
                        "(protocol v2: negative - positive flips)")
    b.add_argument("--no-truth", action="store_true", help="no truth arm")
    b.add_argument("--increment-sources", choices=S.SOURCE_MODES, default=S.SOURCES_RELEVANCE,
                   help="the relevance criterion of V* and OTHER_HEAVY: 'relevance' (default: the relevance file "
                        "below) or 'evidence' (only sources with --min-evidence verified cwd12-species boxes; no "
                        "relevance file)")
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
              increment_sources=a.increment_sources, min_evidence=a.min_evidence)
    except (P.PilotError, D.DriverError, S.SelectError) as e:
        print("[inc.realloop] ERROR: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
