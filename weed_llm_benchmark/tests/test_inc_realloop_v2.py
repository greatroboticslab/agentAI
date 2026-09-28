#!/usr/bin/env python3
"""realloop_v2: `realloop build --increment-sources recovered --step1-overlay`
(docs/FUNNEL_AUDIT.md section 9; docs/FUNNEL_AUDIT_RUNNER.md 5.6.3) and the
recovery provenance of `pilot build-baseline` (runner 5.6.4).

The Step 1 world is tests/test_inc_realloop.py's (make_world: verify's files,
select's build, base B of 220 images, the never-train index and LOCK.json),
imported, not copied. On top of it this test writes a synthetic recovery
overlay INC_DIR/step1_r1/ in the runner's format (section 4.18): recovery.json
(funnel-recovery/1, header inputs, guards, domain_dev, recovered_pool),
recovered_pool.jsonl, content-addressed overlay labels and masked PNG copies
(mean-RGB fill), domain_dev.jsonl and, for the baselines, arms/*.jsonl with
arms.json.
  * Pools (M = 12 requested; floor 0.05 x 220 = 11): VETO 16 images (R-V,
    masked), AUTH 28 (R-A), CLASS 30 (R-C), JUDGE 16 (R-J, masked) and one
    FETCH row (a refetched image outside verify's pool: checked, never drawn).
    Four harvU1 images are held out as the H10d domain dev.
  * near_dup3 groups planted through pool_meta.jsonl dHashes (the admit and
    select summaries re-stamped, so only the checks under test can object):
    AUTH {u1_008, u1_009, u1_010} and {u1_012, u1_013}; CLASS {u2_000 ..
    u2_003}; JUDGE {u2_030, u2_031}; and one group that spans VETO (u1_000)
    and AUTH (u1_020), left out of the draw. AUTH also holds u1_near_4 (1
    dHash bit from an image of base B in test_inc_realloop's world) and
    u1_035 (planted 1 bit from the domain-dev image u1_036): both left out.
  * recovery.json gives every row's stratum a passed gate, and its guard
    record covers exactly the rows (every unmasked original, every masked
    copy).

Pinned:
  * the constants: sequence REC-VETO, REC-AUTH-1, REC-CLASS-1, REC-JUDGE-1,
    REC-AUTH-2, REC-CLASS-2; the pools; the substitution order; the modes;
  * plan_recovered on synthetic pools with |B| = 3,927: all arms full (M 287,
    nothing dropped); JUDGE below the floor (dropped, drawn from the arm with
    the most remaining capacity); AUTH short by one step (capacity per step
    under the floor: dropped, both steps substituted; at 450 images kept and M
    shrinks to 225); the floor 196.35 (M 196 refuses, 197 plans); ties to the
    earlier arm of AUTH, CLASS, VETO; no substitute, or no arm kept, refuses;
  * draw_recovered: exactly M, whole groups, the rule recomputed here
    (groups sorted by first key, default_rng(stable_int(exp + '/' + step))
    permutation, overshooting groups skipped), deterministic; refuses when M
    cannot be reached;
  * the build on the world: 6 steps of M images, every one unclean, kind
    'recovered', pool, policy counts, no substitution; disjoint from the base
    and from each other; near_dup3 groups whole; no step holds a group near a
    base image, near a domain-dev image or spanning two pools (the left-out
    groups recomputed here by brute force over dHash bits); each step's keys
    are the rule's draw recomputed here from the overlay; the manifest rows are the
    overlay rows' manifest keys; exp.json's increment_sources block and
    build_summary.json's "recovered" record (plan, per-step counts, guard,
    the cross-pool group); driver init (3 base + 6 x 3 truth runs, every
    truth 'with' = B + D_k: T never grows); driven to done by a synthetic
    executor, the truth arm decides every step and the report builds;
  * the JUDGE arm below the floor in a real build: dropped, REC-JUDGE-1 drawn
    from VETO (the most remaining capacity), recorded in exp.json and the
    summary; M 10 < floor 11 refuses;
  * every load_overlay refusal on its own planted defect, before anything is
    written: status not complete; testing recovery for a production build;
    a guard record that does not cover every masked copy or every unmasked
    original; a drawable row naming no stratum, or a stratum whose gate did
    not pass; a null near_dup3; a domain-dev key that is not a pool image;
    a stale recorded input; guard hits; changed source labels; no domain dev
    record; recovered_pool.jsonl not the recorded one; a row lacking a key;
    a doubled key; an unknown pool; a quarantined source; a domain-dev image;
    a base image; an unmasked original that is not the pool image; one that
    does not hash as recorded; a wrong dhash_unmasked; an unmasked original
    within 6 bits of an evaluation image (its masked copy clears the guard,
    so it is the unmasked check that refuses); split near_dup3 groups; an
    overlay label that does not hash as recorded; a masked copy within 6 bits
    of an evaluation image;
  * the flags: recovered with --relevance, --min-evidence, --n-verified or
    --no-truth, or without --step1-overlay or --size, refuses; --step1-overlay
    without the mode refuses; the CLI builds (exit 0) and refuses (exit 1);
  * realloop_v1's exp.json (the local copy) still validates as a driver
    definition;
  * build-baseline: an arm manifest named in arms.json (U with 5 seeds, and
    U_ctl with join labels on unmasked images) builds and records
    recovery_build; a manifest no arm names whose recovered rows are
    recovered_pool rows builds in production; one overlay label edited (the
    manifest re-stamped) refuses in production and is recorded in testing;
    a production build of a manifest whose masked copy's unmasked original is
    near an evaluation image refuses; an arm named by a testing arms.json, by
    one whose recorded inputs changed or by one recording no inputs refuses
    in production, and a current one builds; a recovery manifest whose
    recovery.json inputs changed, or whose recovery.json is a testing or an
    incomplete one, refuses in production; a domain-dev image refuses in any
    build; base B's baseline summary has no recovery_build key.

tests/test_inc_realloop.py must still pass unchanged (DEFAULT_BUILD_DIGEST).
No network, no GPU.

Run:  python3 tests/test_inc_realloop_v2.py
"""
import collections
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc_realloop as W0  # noqa: E402  (sets INC_DIR/REPO to its own temporary tree first)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import driver as D  # noqa: E402
from weed_optimizer_framework.tools.inc import gate as G  # noqa: E402
from weed_optimizer_framework.tools.inc import pilot as P  # noqa: E402
from weed_optimizer_framework.tools.inc import realloop as RL  # noqa: E402
from weed_optimizer_framework.tools.inc import report as R  # noqa: E402
from weed_optimizer_framework.tools.inc import select as S  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402

TMP = W0.TMP
FAILURES = []
SKIPS = []
PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL_INC = PKG_ROOT / "results" / "framework" / "inc"


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def raises(fn, exc=Exception, contains=None):
    try:
        fn()
    except exc as e:
        if contains and contains not in str(e):
            print("       raised %s without %r: %s" % (type(e).__name__, contains, e))
            return False
        return True
    except Exception as e:                    # noqa: BLE001 - the wrong exception is a failure, shown
        print("       raised %s, not %s: %s" % (type(e).__name__, exc.__name__, e))
        return False
    print("       did not raise")
    return False


# ------------------------------------------------------------ the overlay
R1 = pathlib.Path(C.INC_DIR) / "step1_r1"
M_REQ = 12
POLICY = {"VETO": "R-V", "AUTH": "R-A", "CLASS": "R-C", "JUDGE": "R-J", "FETCH": "R-F"}
MASKED_POOLS = ("VETO", "JUDGE")
HELD = ["u1_%03d" % j for j in range(36, 40)]


NEAR_BASE = "u1_near_4"          # test_inc_realloop's world: 1 dHash bit from an image of base B
NEAR_DEV = "u1_035"              # planted here 1 bit from the domain-dev image HELD[0]


def default_pools():
    return {"VETO": ["u3_%03d" % j for j in range(8)] + ["u1_%03d" % j for j in range(8)],
            "AUTH": ["u1_%03d" % j for j in range(8, 36)] + [NEAR_BASE],
            "CLASS": ["u2_%03d" % j for j in range(30)],
            "JUDGE": ["u2_%03d" % j for j in range(30, 46)]}


PLANTED = {"AUTH_A": ["u1_008", "u1_009", "u1_010"], "AUTH_B": ["u1_012", "u1_013"],
           "CLASS_A": ["u2_000", "u2_001", "u2_002", "u2_003"], "JUDGE_A": ["u2_030", "u2_031"],
           "CROSS": ["u1_000", "u1_020"]}


def read_meta():
    return [json.loads(ln) for ln in pathlib.Path(V.POOL_META).read_text().splitlines() if ln.strip()]


def set_dhash(new):
    """pool_meta.jsonl dHashes {key: h}; the admit and select summaries re-stamped."""
    meta = read_meta()
    for m in meta:
        if m["key"] in new:
            m["dhash"] = int(new[m["key"]])
    W0.write_rows(V.POOL_META, meta)
    W0.restamp_admit()
    W0.restamp_select("pool_meta")


def plant_groups():
    by = {m["key"]: m["dhash"] for m in read_meta()}
    new = {}
    for members in PLANTED.values():
        h0 = int(by[members[0]])
        for i, k in enumerate(members[1:]):
            new[k] = h0 ^ (1 << (7 + 11 * i)) ^ (1 << (3 + 5 * i))       # 2 bits from the first, 4 from each other
    # members 2 bits from the first and at most 4 from each other: one 3-bit group through the first
    new[NEAR_DEV] = int(by[HELD[0]]) ^ (1 << 9)          # a recovered row 1 bit from a domain-dev image
    set_dhash(new)


def _sha16(path):
    return C.sha256_file(path)[:16]


def _content_file(dirpath, stem, suffix, write):
    """Write through write(tmp), then name the file <stem>.<sha16><suffix>."""
    dirpath.mkdir(parents=True, exist_ok=True)
    tmp = dirpath / (".%s.tmp%s" % (stem, suffix))
    write(tmp)
    final = dirpath / ("%s.%s%s" % (stem, _sha16(tmp), suffix))
    os.replace(tmp, final)
    return final


def _mask(src, box, out_dir, stem):
    img = Image.open(src).convert("RGB")
    a = np.asarray(img).copy()
    mean = a.reshape(-1, 3).mean(0).round().astype(np.uint8)
    H, Wd = a.shape[:2]
    _c, cx, cy, w, h = box
    x0, x1 = int(np.floor((cx - w / 2) * Wd)), int(np.ceil((cx + w / 2) * Wd))
    y0, y1 = int(np.floor((cy - h / 2) * H)), int(np.ceil((cy + h / 2) * H))
    a[max(0, y0):min(H, y1), max(0, x0):min(Wd, x1)] = mean
    return _content_file(out_dir, stem, ".png", lambda p: Image.fromarray(a).save(p, format="PNG"))


def fetch_image():
    p = W0.REPO / "funnel_refetch" / "harvF" / "fetch_000.png"
    if not p.exists():
        W0._png(p, np.random.default_rng(99))
    return p


def write_overlay(pools=None, testing=True, fetch=True, held=None, mutate_rows=None, mutate_rec=None,
                  quarantined=("harvQ",), dd_form="document"):
    """(the rows' strata each get a passed gate in recovery.json, and the guard record covers exactly the
    rows: every unmasked original and every masked copy)"""
    """The synthetic step1_r1/ overlay, rebuilt from the current Step 1 files.
    Returns (recovery.json dict, rows)."""
    pools = pools or default_pools()
    held = HELD if held is None else held
    if R1.exists():
        shutil.rmtree(R1)
    R1.mkdir(parents=True)
    pool = {r["key"]: r for r in C.read_manifest(V.POOL)}
    meta = {m["key"]: m for m in read_meta()}
    rows = []
    for pname in sorted(pools):
        for i, key in enumerate(sorted(pools[pname])):
            pr = pool[key]
            boxes = C.read_yolo(pr["label"])
            keep = [(int((b[0] + i) % 12),) + tuple(b[1:]) for b in boxes]
            masked = []
            if pname in MASKED_POOLS:
                masked_box = (C.OTHER_PLANT, 0.25, 0.25, 0.5, 0.5)
                img = _mask(pr["image"], masked_box, R1 / "images_masked" / pr["source"], key)
                masked = [list(masked_box)]
            else:
                img = pathlib.Path(pr["image"])
            lab = _content_file(R1 / "labels_overlay" / pr["source"], key, ".txt",
                                lambda p, bx=keep: C.write_yolo(p, bx))
            rows.append({"image": str(img), "label": str(lab), "sha256": C.sha256_file(img),
                         "label_sha256": C.sha256_file(lab), "source": pr["source"], "session": "", "key": key,
                         "pool": pname, "policy": POLICY[pname], "strata": ["G2/source=%s" % pr["source"]],
                         "unmasked_image": pr["image"], "unmasked_sha256": pr["sha256"], "masked_boxes": masked,
                         "ctl_label": pr["label"], "ctl_label_sha256": pr["label_sha256"],
                         "provenance_group": "src:%s" % pr["source"], "lab": "src:%s" % pr["source"],
                         "licence": "CC BY 4.0", "dhash_unmasked": int(meta[key]["dhash"]),
                         "dhash_masked": C.dhash(str(img))})
    if fetch:
        img = fetch_image()
        lab = _content_file(R1 / "labels_overlay" / "harvF", "fetch_000", ".txt",
                            lambda p: C.write_yolo(p, [(10, 0.5, 0.5, 0.2, 0.2)]))
        rows.append({"image": str(img), "label": str(lab), "sha256": C.sha256_file(img),
                     "label_sha256": C.sha256_file(lab), "source": "harvF", "session": "", "key": "fetch_000",
                     "pool": "FETCH", "policy": "R-F", "strata": [], "unmasked_image": str(img),
                     "unmasked_sha256": C.sha256_file(img), "masked_boxes": [], "ctl_label": str(lab),
                     "ctl_label_sha256": C.sha256_file(lab), "provenance_group": "src:harvF", "lab": "src:harvF",
                     "licence": "CC BY 4.0", "dhash_unmasked": C.dhash(str(img)), "dhash_masked": C.dhash(str(img))})
    rows.sort(key=lambda r: r["key"])
    grp = S.dup_groups([r["dhash_unmasked"] for r in rows], bits=3)
    first = {}
    for g, r in zip(grp, rows):
        first.setdefault(int(g), r["key"])
    for g, r in zip(grp, rows):
        r["near_dup3"] = "n:%s" % first[int(g)]
    if mutate_rows:
        rows = mutate_rows(rows) or rows
    W0.write_rows(R1 / "recovered_pool.jsonl", rows)
    dd = [{f: pool[k][f] for f in C.MANIFEST_KEYS} if k in pool else {"key": k} for k in held]
    W0.write_rows(R1 / "domain_dev.jsonl", dd)
    dd_sha = C.sha256_file(R1 / "domain_dev.jsonl")
    (R1 / "domain_dev.json").write_text(json.dumps(
        {"format": "funnel-domain-dev/1", "min_group_images": 20, "sources": {},
         "rows": {"path": str(R1 / "domain_dev.jsonl"), "sha256": dd_sha, "images": len(dd)}}, indent=1))
    if dd_form == "document":        # the recover module's form: the document, and the rows' sha256 beside it
        dd_record = {"path": str(R1 / "domain_dev.json"), "sha256": C.sha256_file(R1 / "domain_dev.json"),
                     "rows_sha256": dd_sha}
    else:                            # the runner's section 4.18 form: the rows themselves
        dd_record = {"path": str(R1 / "domain_dev.jsonl"), "sha256": dd_sha}

    def rec_(p):
        return {"path": str(p), "sha256": C.sha256_file(p), "bytes": os.path.getsize(p)}
    n = len(rows)
    n_masked = sum(1 for r in rows if r["image"] != r["unmasked_image"])
    gates = {s: {"policy": r["policy"], "level": "species", "n_labelled": 30, "box_gate": True, "class_gate": None,
                 "precision": {"estimate": 0.95, "lb": 0.88, "ub": 0.99, "rogan_gladen": True}, "passed": True}
             for r in rows for s in r["strata"]}
    rec = {"format": "funnel-recovery/1", "domain": "weed", "built_utc": "2026-09-28T00:00:00Z",
           "prereg": {"path": "x", "sha256": "0" * 64, "core_sha256": "0" * 64},
           "contract": {"path": "docs/FUNNEL_AUDIT.md", "sha256": "0" * 64},
           "domain_config": {"path": "x", "sha256": "0" * 64}, "code": {},
           "inputs": {"pool_meta": rec_(V.POOL_META), "pool": rec_(V.POOL), "base_B": rec_(W0.base_b())},
           "seeds": {}, "testing": bool(testing), "status": "complete", "refusals": [],
           "gates": gates, "class_maps": [], "identity_checks": {}, "quarantined_sources": list(quarantined),
           "guards": {"never_train": {"unmasked_checked": n, "masked_checked": n_masked, "hits": 0, "unhashable": 0},
                      "h6": {"unmasked_checked": n, "masked_checked": n_masked, "copies": 0}},
           "counts": {}, "licences": {}, "provenance_groups": {},
           "source_labels_unchanged": {"checked": n, "changed": 0},
           "recovered_pool": {"path": str(R1 / "recovered_pool.jsonl"),
                              "sha256": C.sha256_file(R1 / "recovered_pool.jsonl")},
           "domain_dev": dd_record}
    if mutate_rec:
        rec = mutate_rec(rec) or rec
    (R1 / "recovery.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    return rec, rows


def not_built(exp):
    return W0.not_built(exp) and not D.Paths(exp).manifests.exists()


def build(exp, **kw):
    kw = dict({"replay_mode": "full", "recipes": "full", "testing": True, "quiet": True,
               "increment_sources": "recovered", "step1_overlay": R1, "size": M_REQ, "init": False}, **kw)
    return RL.build(exp, base=W0.base_b(), **kw)


def refuses(exp, contains, **kw):
    return raises(lambda: build(exp, **kw), RL.RealLoopError, contains) and not_built(exp)


# ------------------------------------------------------------------ units
def groups_of(sizes, prefix):
    out, j = [], 0
    for n in sizes:
        out.append([{"key": "%s_%05d" % (prefix, j + i)} for i in range(n)])
        j += n
    return out


def test_units():
    print("constants, plan_recovered and draw_recovered")
    check("the recovered sequence, its pools, the substitution order and the modes",
          RL.RECOVERED_SEQUENCE == ("REC-VETO", "REC-AUTH-1", "REC-CLASS-1", "REC-JUDGE-1", "REC-AUTH-2",
                                    "REC-CLASS-2")
          and [RL.RECOVERED_POOLS[s] for s in RL.RECOVERED_SEQUENCE] == ["VETO", "AUTH", "CLASS", "JUDGE", "AUTH",
                                                                          "CLASS"]
          and RL.SUBSTITUTION_ORDER == ("AUTH", "CLASS", "VETO")
          and RL.INCREMENT_SOURCE_MODES == S.SOURCE_MODES + ("recovered",) and RL.KIND_RECOVERED == "recovered")
    B = 3927

    def pools(veto, auth, clas, judge):
        return {"VETO": groups_of([1] * veto, "v"), "AUTH": groups_of([1] * auth, "a"),
                "CLASS": groups_of([1] * clas, "c"), "JUDGE": groups_of([1] * judge, "j")}
    p = RL.plan_recovered(pools(500, 1200, 1000, 400), 287, B)
    check("all arms full: M 287, nothing dropped or substituted, every step from its own arm",
          p["m"] == 287 and not p["dropped"] and not p["substituted"]
          and p["steps"] == dict(RL.RECOVERED_POOLS) and abs(p["floor"] - 196.35) < 1e-9, p)
    p = RL.plan_recovered(pools(500, 1200, 1000, 150), 287, B)
    check("JUDGE below the floor (150 per step < 196.35): dropped; REC-JUDGE-1 from AUTH (remaining 626 > CLASS "
          "426 > VETO 213)",
          p["m"] == 287 and [d["pool"] for d in p["dropped"]] == ["JUDGE"]
          and p["substituted"] == [{"step": "REC-JUDGE-1", "from": "JUDGE", "to": "AUTH", "remaining_before": 626}]
          and p["steps"]["REC-JUDGE-1"] == "AUTH", p)
    p = RL.plan_recovered(pools(500, 300, 1500, 400), 287, B)
    check("AUTH short by one step (300 images: 150 per step < the floor): dropped, both steps from CLASS, the most "
          "remaining (926, then 639)",
          p["m"] == 287 and [d["pool"] for d in p["dropped"]] == ["AUTH"]
          and [(x["step"], x["to"], x["remaining_before"]) for x in p["substituted"]]
          == [("REC-AUTH-1", "CLASS", 926), ("REC-AUTH-2", "CLASS", 639)], p)
    p = RL.plan_recovered(pools(500, 450, 1500, 400), 287, B)
    check("AUTH at 450 images (225 per step >= the floor): kept, and M shrinks to 225 for every step",
          p["m"] == 225 and not p["dropped"] and not p["substituted"] and p["m_requested"] == 287, p)
    check("the floor is 0.05 x |B| as a real number: M 196 < 196.35 refuses, 197 plans",
          raises(lambda: RL.plan_recovered(pools(500, 1200, 1000, 400), 196, B), RL.RealLoopError, "below the floor")
          and RL.plan_recovered(pools(500, 1200, 1000, 400), 197, B)["m"] == 197)
    p = RL.plan_recovered(pools(500 + 287, 574 + 500, 574 + 500, 10), 287, B)
    check("a tie in remaining capacity goes to the earlier arm of AUTH, CLASS, VETO",
          p["substituted"][0]["to"] == "AUTH", p["substituted"])
    check("no arm with M images left for a dropped arm's step refuses",
          raises(lambda: RL.plan_recovered(pools(300, 600, 600, 10), 287, B), RL.RealLoopError, "no substitute"))
    check("no arm at the floor refuses", raises(lambda: RL.plan_recovered(pools(10, 10, 10, 10), 287, B),
                                                RL.RealLoopError, "no recovered arm"))

    gs = groups_of([3, 1, 5, 2, 1, 4, 2, 1, 1, 6], "g")
    got = RL.draw_recovered("exp_x", "REC-AUTH-1", gs, 9)
    srt = sorted(gs, key=lambda g: g[0]["key"])
    order = np.random.default_rng(C.stable_int("exp_x/REC-AUTH-1")).permutation(len(srt))
    want, n = [], 0
    for i in order:
        if n + len(srt[int(i)]) <= 9:
            want += srt[int(i)]
            n += len(srt[int(i)])
            if n == 9:
                break
    gid = {r["key"]: i for i, g in enumerate(gs) for r in g}
    whole = all(sum(1 for r in got if gid[r["key"]] == i) in (0, len(g)) for i, g in enumerate(gs))
    check("draw_recovered: exactly M from whole groups, the rule's draw recomputed here, deterministic",
          len(got) == 9 and whole and [r["key"] for r in got] == sorted(r["key"] for r in want)
          and got == RL.draw_recovered("exp_x", "REC-AUTH-1", list(reversed(gs)), 9))
    check("draw_recovered refuses when whole groups cannot reach M",
          raises(lambda: RL.draw_recovered("exp_x", "REC-VETO", groups_of([5, 5], "h"), 3), RL.RealLoopError,
                 "not 3"))


# ------------------------------------------------------------- the build
def excluded_groups(rows):
    """{near_dup3: reasons} of the groups the draw must leave out, by brute force here: a row within 3 dHash
    bits of a base image of verify's pool or of a domain-dev image, or a group in more than one pool."""
    meta = {m["key"]: int(m["dhash"]) for m in read_meta()}
    base = [meta[r["key"]] for r in C.read_manifest(W0.base_b()) if r["key"] in meta]
    dev = [meta[r["key"]] for r in C.read_manifest(R1 / "domain_dev.jsonl")]
    out = collections.defaultdict(set)
    pools = collections.defaultdict(set)
    for r in rows:
        pools[r["near_dup3"]].add(r["pool"])
        h = int(r["dhash_unmasked"])
        if any(W0.bits(h, b) <= 3 for b in base):
            out[r["near_dup3"]].add("near_base")
        if any(W0.bits(h, b) <= 3 for b in dev):
            out[r["near_dup3"]].add("near_domain_dev")
    for g, p in pools.items():
        if len(p) > 1:
            out[g].add("cross_pool")
    return dict(out)


def recompute(exp, rows, plan):
    """Each step's keys by the rule, recomputed here from the overlay rows."""
    gone = excluded_groups(rows)
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["pool"] != "FETCH" and r["near_dup3"] not in gone:
            by[r["pool"]][r["near_dup3"]].append(r["key"])
    rem = {p: sorted((sorted(g) for g in gs.values()), key=lambda g: g[0]) for p, gs in by.items()}
    out = {}
    for s in RL.RECOVERED_SEQUENCE:
        p = plan["steps"][s]
        groups = sorted(rem[p], key=lambda g: g[0])
        order = np.random.default_rng(C.stable_int(exp + "/" + s)).permutation(len(groups))
        take, n = [], 0
        for i in order:
            g = groups[int(i)]
            if n + len(g) <= plan["m"]:
                take += g
                n += len(g)
                if n == plan["m"]:
                    break
        out[s] = sorted(take)
        rem[p] = [g for g in groups if g[0] not in set(take)]
    return out


def test_build():
    print("realloop build --increment-sources recovered")
    rec, rows = write_overlay()
    base_rows = C.read_manifest(W0.base_b())
    exp = "rv2_t"
    fb = D.FakeBackend()
    summary, defn, _ = build(exp, init=True, backend=fb, recipes="full,lora")
    paths = D.Paths(exp)
    seq = list(RL.RECOVERED_SEQUENCE)
    steps = {s["name"]: s for s in defn["steps"]}
    got = {s: C.read_manifest(steps[s]["manifest"]) for s in seq}
    check("exp.json and build_summary.json are written; each step's manifest is manifests/<step>.jsonl",
          paths.exp_json.is_file() and (paths.root / P.BUILD_SUMMARY).is_file()
          and all(steps[s]["manifest"] == str(paths.manifests / ("%s.jsonl" % s)) for s in seq))
    check("six recovered steps of M = 12 images, every one unclean, kind 'recovered', from its own pool, no "
          "substitution",
          [s["name"] for s in defn["steps"]] == seq and defn["increment_images"] == 12
          and all(len(got[s]) == 12 and steps[s]["n_images"] == 12 and steps[s]["clean"] is False
                  and steps[s]["kind"] == "recovered" and steps[s]["pool"] == RL.RECOVERED_POOLS[s]
                  and steps[s]["substituted_from"] is None for s in seq)
          and summary["clean"] == [] and defn["truth"] is True, [(s["name"], s["n_images"]) for s in defn["steps"]])
    plan = summary["recovered"]["plan"]
    want = recompute(exp, rows, plan)
    check("each step's keys are the rule's draw recomputed here from recovered_pool.jsonl",
          all([r["key"] for r in got[s]] == want[s] for s in seq), {s: [r["key"] for r in got[s]][:3] for s in seq})
    by_key = {r["key"]: r for r in rows}
    check("the manifest rows are the overlay rows' manifest keys (masked image, overlay label), and FETCH is never "
          "drawn",
          all(r == {k: by_key[r["key"]][k] for k in C.MANIFEST_KEYS} for s in seq for r in got[s])
          and not any(by_key[r["key"]]["pool"] == "FETCH" for s in seq for r in got[s])
          and summary["recovered"]["pools"]["FETCH"]["drawn"] is False)
    parts = [("base", base_rows)] + [(s, got[s]) for s in seq]
    clash = [(a, b, f) for i, (a, ra) in enumerate(parts) for b, rb in parts[:i]
             for f in ("key", "image", "sha256") if {r[f] for r in ra} & {r[f] for r in rb}]
    check("the base and every step pairwise disjoint (key, image path, sha256)", not clash, clash[:3])
    step_of = {r["key"]: s for s in seq for r in got[s]}
    split = [name for name, members in PLANTED.items() if len({step_of.get(k) for k in members}) != 1]
    check("planted near_dup3 groups stay whole (all of a group in one step, or none of it drawn)", not split, split)
    gone = excluded_groups(rows)
    left_out = sorted(r["key"] for r in rows if r["near_dup3"] in gone)
    check("no step draws a group near a base image (%s), near a domain-dev image (%s) or spanning two pools "
          "(%s)" % (NEAR_BASE, NEAR_DEV, PLANTED["CROSS"]),
          sorted(left_out) == sorted([NEAR_BASE, NEAR_DEV] + PLANTED["CROSS"])
          and not any(k in step_of for k in left_out)
          and {g: sorted(v) for g, v in gone.items()} == {"n:%s" % NEAR_BASE: ["near_base"],
                                                         "n:%s" % NEAR_DEV: ["near_domain_dev"],
                                                         "n:%s" % PLANTED["CROSS"][0]: ["cross_pool"]},
          (left_out, gone))
    check("policy counts per step follow the pool (R-V, R-A, R-C, R-J)",
          all(steps[s]["policy_counts"] == {POLICY[RL.RECOVERED_POOLS[s]]: 12} for s in seq))
    inc = defn["step1"]["increment_sources"]
    check("exp.json step1.increment_sources: mode recovered, the overlay's sha256s, the rule, M requested and M",
          inc["mode"] == "recovered" and inc["m_requested"] == 12 and inc["m"] == 12 and inc["dropped"] == []
          and inc["substituted"] == [] and inc["rule"] == RL.RECOVERED_RULE
          and inc["overlay"] == {"dir": str(R1), "recovery_sha256": C.sha256_file(R1 / "recovery.json"),
                                 "recovered_pool_sha256": C.sha256_file(R1 / "recovered_pool.jsonl")}
          and summary["step1"] == defn["step1"], inc)
    rs = summary["recovered"]
    check("build_summary.json 'recovered': the plan's capacities, per-step counts, the guard record, the domain "
          "dev and the cross-pool near_dup3 group",
          rs["plan"]["capacity"]["AUTH"] == {"images": 26, "groups": 23, "steps": 2, "per_step": 13.0}
          and rs["pools"]["AUTH"]["images"] == 29 and rs["pools"]["AUTH"]["eligible_images"] == 26
          and rs["excluded"]["groups"] == 3 and rs["excluded"]["images"] == 4
          and rs["excluded"]["by_reason"] == {"near_base": {"groups": 1, "images": 1, "by_pool": {"AUTH": 1}},
                                              "near_domain_dev": {"groups": 1, "images": 1, "by_pool": {"AUTH": 1}},
                                              "cross_pool": {"groups": 1, "images": 2,
                                                             "by_pool": {"AUTH": 1, "VETO": 1}}}
          and rs["plan"]["floor"] == 11.0 and all(rs["per_step"][s]["images"] == 12 for s in seq)
          and rs["guard"]["unmasked"] == {"checked": len(rows), "hits": 0, "unhashable": 0,
                                          "dhash_from": {"pool_meta": len(rows) - 1, "file": 1}}
          and rs["guard"]["trained"]["hits"] == 0 and rs["overlay"]["domain_dev"]["rows"]["images"] == len(HELD)
          and rs["overlay"]["domain_dev"]["path"].endswith("domain_dev.json")
          and rs["overlay"]["inputs_checked"] == 3
          and rs["cross_pool_near_dup3"]["groups"] == 1
          and list(rs["cross_pool_near_dup3"]["first"].values()) == [["AUTH", "VETO"]], rs["plan"]["capacity"])
    check("the recovered attribution scope and the steps' boxes",
          defn["attribution_scope"] == RL.RECOVERED_ATTRIBUTION_SCOPE
          and all(summary["increments"][s]["boxes"] == C.class_counts(got[s]) for s in seq))

    specs = [json.loads(pathlib.Path(p).read_text()) for p in fb.submissions[0]["specs"]]
    union = {s["run_id"]: s for s in specs if s["kind"] == "union"}
    bk = {r["key"] for r in base_rows}
    ok = all(set(W0.keys_of(union["truth__s%02d_%s__union__s%d" % (i, s, seed)]["train_manifest"]))
             == bk | {r["key"] for r in got[s]} for i, s in enumerate(seq, 1) for seed in (0, 1, 2))
    check("driver init: 3 base + 6 x 3 truth runs; every truth 'with' set is B + D_k (T never grows)",
          len([s for s in specs if s["kind"] == "base"]) == 3 and len(union) == 18 and ok)
    ex = W0.Executor({s: {r["key"] for r in got[s]} for s in seq}, set())
    st = W0.drive(exp, fb, ex)
    truths = {e["step"]: e["detail"]["verdict"] for e in W0.ledger_of(exp) if e["type"] == "truth"}
    rep = R.build(exp)
    check("driven to done: the truth arm decides every step against B's runs, and the report builds",
          st["done"] and truths == {s: G.HELPS for s in seq} and len(rep["steps"]) == 6, truths)
    return rows


def test_substitution():
    print("a real build with JUDGE below the floor, and the floor")
    pools = default_pools()
    pools["VETO"] = pools["VETO"] + ["u2_%03d" % j for j in range(46, 60)]
    pools["JUDGE"] = ["u2_%03d" % j for j in range(30, 40)]
    write_overlay(pools, dd_form="rows")          # the domain dev record naming the rows file itself
    summary, defn, _ = build("rv2_sub")
    steps = {s["name"]: s for s in defn["steps"]}
    inc = defn["step1"]["increment_sources"]
    check("JUDGE (10 images < floor 11) is dropped; REC-JUDGE-1 is drawn from VETO, the most remaining capacity "
          "(17 > CLASS 6 > AUTH 2, after the near-duplicate groups are left out), and exp.json records it",
          [d["pool"] for d in inc["dropped"]] == ["JUDGE"]
          and inc["substituted"] == [{"step": "REC-JUDGE-1", "from": "JUDGE", "to": "VETO", "remaining_before": 17}]
          and steps["REC-JUDGE-1"]["pool"] == "VETO" and steps["REC-JUDGE-1"]["substituted_from"] == "JUDGE"
          and summary["recovered"]["per_step"]["REC-JUDGE-1"]["substituted_from"] == "JUDGE"
          and steps["REC-JUDGE-1"]["policy_counts"] == {"R-V": 12}, inc)
    check("a domain_dev record that names the rows file itself (the runner's 4.18 form) is read as well",
          summary["recovered"]["overlay"]["domain_dev"]["path"].endswith("domain_dev.jsonl")
          and summary["recovered"]["overlay"]["domain_dev"]["rows"]["images"] == len(HELD))
    write_overlay()
    check("M 10 is below the floor 11 (0.05 x 220): refuses before anything is written",
          refuses("rv2_floor", "below the floor", size=10))


def test_refusals():
    print("load_overlay refusals, each before anything is written")
    cases = []

    def case(name, contains, mutate_rows=None, mutate_rec=None, after=None, **kw):
        cases.append((name, contains, mutate_rows, mutate_rec, after, kw))

    def drop_key(rows):
        del rows[0]["near_dup3"]

    def dup_key(rows):
        rows.append(dict(rows[-1]))

    def bad_pool(rows):
        rows[0]["pool"] = "OTHER"

    def base_row(rows):
        b = [r for r in C.read_manifest(W0.base_b()) if r["key"].startswith("pool_")][0]
        pr = {r["key"]: r for r in C.read_manifest(V.POOL)}[b["key"]]
        rows.append(dict(rows[0], image=b["image"], sha256=b["sha256"], label=b["label"],
                         label_sha256=b["label_sha256"], key=b["key"], source=b["source"], unmasked_image=pr["image"],
                         unmasked_sha256=pr["sha256"], near_dup3="n:%s" % b["key"],
                         dhash_unmasked=int({m["key"]: m for m in read_meta()}[b["key"]]["dhash"])))

    def wrong_unmasked(rows):
        r = next(r for r in rows if r["key"] == "u1_020")
        other = {x["key"]: x for x in C.read_manifest(V.POOL)}["u1_021"]
        r["unmasked_image"], r["unmasked_sha256"] = other["image"], other["sha256"]

    def fetch_changed(rows):
        r = next(r for r in rows if r["pool"] == "FETCH")
        r["unmasked_sha256"] = "0" * 64

    def wrong_dhash(rows):
        rows[3]["dhash_unmasked"] = int(rows[3]["dhash_unmasked"]) ^ 1

    def split_group(rows):
        for r in rows:
            if r["key"] == "u1_009":
                r["near_dup3"] = "n:u1_009"

    def quarantined_row(rows):
        next(r for r in rows if r["key"] == "u1_020")["source"] = "harvQ"

    def held_row(rows):
        pool = {x["key"]: x for x in C.read_manifest(V.POOL)}
        meta = {m["key"]: m for m in read_meta()}
        k = HELD[0]
        rows.append(dict(rows[-1], image=pool[k]["image"], sha256=pool[k]["sha256"], key=k,
                         source=pool[k]["source"], unmasked_image=pool[k]["image"], unmasked_sha256=pool[k]["sha256"],
                         near_dup3="n:%s" % k, dhash_unmasked=int(meta[k]["dhash"])))

    def no_strata(rows):
        next(r for r in rows if r["pool"] == "VETO")["strata"] = []

    def gate_failed(rec):
        s = sorted(rec["gates"])[0]
        rec["gates"][s] = dict(rec["gates"][s], passed=False)

    case("status not complete", "not 'complete'", mutate_rec=lambda r: dict(r, status="refused"))
    case("a guard record that does not cover every masked copy", "do not cover",
         mutate_rec=lambda r: dict(r, guards=dict(r["guards"], h6=dict(r["guards"]["h6"], masked_checked=0))))
    case("a guard record that does not cover every unmasked original", "do not cover",
         mutate_rec=lambda r: dict(r, guards=dict(r["guards"], never_train=dict(
             r["guards"]["never_train"], unmasked_checked=r["guards"]["never_train"]["unmasked_checked"] - 1))))
    case("a drawable row that names no stratum", "name no stratum", mutate_rows=no_strata)
    case("a row whose near_dup3 group is null", "no near_dup3 group",
         mutate_rows=lambda rows: rows[2].update(near_dup3=None))
    case("a row whose stratum's gate did not pass", "does not record as passed", mutate_rec=gate_failed)
    case("a domain-dev key that is not a pool image", "not verify's pool images", held=HELD + ["zz_not_in_pool"])
    case("a testing recovery for a production build", "testing recover",
         kwargs_prod=True)
    case("guard hits recorded", "each must be 0",
         mutate_rec=lambda r: dict(r, guards=dict(r["guards"], h6={"copies": 2})))
    case("changed source labels recorded", "changed source label",
         mutate_rec=lambda r: dict(r, source_labels_unchanged={"checked": 3, "changed": 1}))
    case("no domain dev record", "domain_dev", mutate_rec=lambda r: {k: v for k, v in r.items() if k != "domain_dev"})
    case("domain-dev rows that changed after recover", "hashes to",
         after=lambda: (R1 / "domain_dev.jsonl").write_text((R1 / "domain_dev.jsonl").read_text() + "\n"))
    case("a domain-dev rows_sha256 that is not the rows'", "rows_sha256",
         mutate_rec=lambda r: dict(r, domain_dev=dict(r["domain_dev"], rows_sha256="0" * 64)))
    case("recovered_pool.jsonl not the recorded one", "not the one recover wrote",
         after=lambda: (R1 / "recovered_pool.jsonl").write_text((R1 / "recovered_pool.jsonl").read_text() + "\n"))
    case("a row lacking an overlay key", "lack some of", mutate_rows=drop_key)
    case("a doubled key", "twice", mutate_rows=dup_key)
    case("an unknown pool", "are not among", mutate_rows=bad_pool)
    case("a quarantined source", "quarantined", mutate_rows=quarantined_row)
    case("a domain-dev image", "domain dev", mutate_rows=held_row)
    case("an image of the base", "images of the base", mutate_rows=base_row)
    case("an unmasked original that is not the pool image", "is not their pool image", mutate_rows=wrong_unmasked)
    case("an unmasked original that does not hash as recorded", "does not hash as recorded",
         mutate_rows=fetch_changed)
    case("a dhash_unmasked that is not pool_meta's", "dhash_unmasked", mutate_rows=wrong_dhash)
    case("near_dup3 groups split within 3 bits", "different near_dup3", mutate_rows=split_group)
    case("an overlay label that does not hash as recorded", "label bytes differ",
         after=lambda: _append(next(iter(sorted((R1 / "labels_overlay").rglob("*.txt"))))))
    for name, contains, mrows, mrec, after, kw in cases:
        prod = kw.pop("kwargs_prod", False)
        with W0.Edited():
            if prod:
                write_overlay(testing=True)
                ok = refuses("rv2_bad", contains, testing=False)
            else:
                write_overlay(mutate_rows=mrows, mutate_rec=mrec, **kw)
                if after:
                    after()
                ok = refuses("rv2_bad", contains)
        check("refuses %s" % name, ok)

    # a stale recorded input: pool_meta.jsonl changed after recover (Step 1 re-stamped, so only this objects)
    with W0.Edited():
        write_overlay()
        meta = read_meta()
        W0.write_rows(V.POOL_META, meta[::-1])
        W0.restamp_admit()
        W0.restamp_select("pool_meta")
        check("refuses a recovery whose recorded input (pool_meta.jsonl) changed after it was written",
              refuses("rv2_bad", "changed after recover"))

    # the never-train guard on the unmasked original, while its masked copy clears the guard
    guard = C.NeverTrainGuard.load()
    with W0.Edited():
        never = json.loads(C.NEVER_TRAIN_INDEX.read_text())["entries"]
        set_dhash({"u3_001": int(never[0][0]) ^ 0b1})
        _rec, rows = write_overlay()
        r = next(x for x in rows if x["key"] == "u3_001")
        hits_masked, _ = guard.check([r["image"]])
        hits_unmasked, _ = guard.check([r["unmasked_image"]], hash_fn=lambda p: r["dhash_unmasked"])
        check("refuses an unmasked original within 6 bits of an evaluation image (pool_meta dHash), although its "
              "masked copy clears the guard",
              not hits_masked and hits_unmasked and refuses("rv2_bad", "unmasked originals"))
    # a masked copy within 6 bits of an evaluation image (inc/train.py's check and the guard)
    with W0.Edited():
        _rec, rows = write_overlay()
        r = next(x for x in rows if x["key"] == "u2_035")
        data = json.loads(C.NEVER_TRAIN_INDEX.read_text())
        data["entries"].append([C.dhash(r["image"]) ^ 0b10, "dev", "/x/planted_eval.jpg"])
        data["min_expected"] += 1
        C.NEVER_TRAIN_INDEX.write_text(json.dumps(data))
        check("refuses a masked copy within 6 bits of an evaluation image (check_training_manifest's guard)",
              refuses("rv2_bad", "never-train"))


def _append(path):
    with open(path, "a") as fh:
        fh.write("12 0.500000 0.500000 0.100000 0.100000\n")


def test_flags():
    print("flags and the CLI")
    write_overlay()
    for name, contains, kw in (
            ("--relevance", "refuses --relevance", {"relevance": TMP / "rel.json"}),
            ("--min-evidence", "--min-evidence", {"min_evidence": 1}),
            ("--n-verified", "--n-verified", {"n_verified": 6}),
            ("--no-truth", "--no-truth", {"truth": False}),
            ("no --step1-overlay", "needs --step1-overlay", {"step1_overlay": None}),
            ("no --size", "needs --size", {"size": None})):
        check("recovered refuses %s" % name, refuses("rv2_flag", contains, **kw))
    check("--step1-overlay without --increment-sources recovered refuses",
          raises(lambda: RL.build("rv2_flag2", base=W0.base_b(), replay_mode="full", recipes="full", testing=True,
                                  init=False, quiet=True, step1_overlay=R1), RL.RealLoopError,
                 "applies to --increment-sources recovered") and not_built("rv2_flag2"))
    fb = D.FakeBackend()
    saved = D.SlurmBackend
    D.SlurmBackend = lambda *a, **k: fb
    try:
        rc = RL.main(["build", "--exp", "rv2_cli", "--replay-mode", "full", "--recipes", "full", "--gate-flips-mode",
                      "net", "--increment-sources", "recovered", "--step1-overlay", str(R1), "--size", str(M_REQ),
                      "--testing", "--quiet"])
        rc_bad = RL.main(["build", "--exp", "rv2_cli2", "--replay-mode", "full", "--recipes", "full",
                          "--step1-overlay", str(R1), "--testing", "--quiet"])
        rc_nv = RL.main(["build", "--exp", "rv2_cli3", "--replay-mode", "full", "--recipes", "full",
                         "--increment-sources", "recovered", "--step1-overlay", str(R1), "--size", "12",
                         "--n-verified", "6", "--testing", "--quiet"])
    finally:
        D.SlurmBackend = saved
    d = json.loads(D.Paths("rv2_cli").exp_json.read_text()) if rc == 0 else {}
    check("CLI: --increment-sources recovered --step1-overlay --size builds and inits (exit 0, gate net); the "
          "overlay without the mode and --n-verified with it exit 1 and build nothing",
          rc == 0 and d["step1"]["increment_sources"]["mode"] == "recovered" and d["gate"] == {"flips_mode": "net"}
          and len(fb.submissions) == 1 and rc_bad == 1 and rc_nv == 1
          and not_built("rv2_cli2") and not_built("rv2_cli3"))

    v1 = REAL_INC / "realloop_v1" / "exp.json"
    if v1.is_file():
        defn = json.loads(v1.read_text())
        ok = True
        try:
            D.validate_definition(defn)
        except D.DriverError as e:
            ok = False
            print("       %s" % e)
        check("realloop_v1's exp.json (the local copy) still validates as a driver definition",
              ok and defn["step1"]["increment_sources"]["mode"] == "evidence")
    else:
        SKIPS.append("realloop_v1 exp.json")
        print("  SKIP realloop_v1's exp.json is not in the local results tree")


# ------------------------------------------------------ build-baseline
def arm_rows(rows, ctl=False):
    """Base B verbatim + the recovered rows (overlay labels and masked copies,
    or with ctl: join labels on the unmasked images)."""
    out = [dict(r) for r in C.read_manifest(W0.base_b())]
    for r in rows:
        if r["pool"] == "FETCH":
            continue
        if ctl:
            out.append({"image": r["unmasked_image"], "label": r["ctl_label"], "sha256": r["unmasked_sha256"],
                        "label_sha256": r["ctl_label_sha256"], "source": r["source"], "session": "", "key": r["key"]})
        else:
            out.append({k: r[k] for k in C.MANIFEST_KEYS})
    return out


def write_arms(rows, testing=False, mutate=None):
    """arms/U.jsonl, U_ctl.jsonl and arms.json with recover --arms' header: format, testing, and the inputs it
    was made from (recovery.json, recovered_pool.jsonl, the base)."""
    arms = {}
    for name, ctl in (("U", False), ("U_ctl", True)):
        p = R1 / "arms" / ("%s.jsonl" % name)
        sha = C.write_manifest(p, arm_rows(rows, ctl))
        arms[name] = {"path": str(p), "sha256": sha, "images": len(C.read_manifest(p))}
    inputs = {n: {"path": str(f), "sha256": C.sha256_file(f), "bytes": os.path.getsize(f)}
              for n, f in (("recovery", R1 / "recovery.json"), ("recovered_pool", R1 / "recovered_pool.jsonl"),
                           ("base", W0.base_b()))}
    doc = {"format": "funnel-arms/1", "testing": bool(testing), "inputs": inputs, "arms": arms}
    if mutate:
        doc = mutate(doc) or doc
    (R1 / "arms" / "arms.json").write_text(json.dumps(doc, indent=1))
    return arms


def test_baseline():
    print("build-baseline on recovery manifests")
    _rec, rows = write_overlay(testing=False)
    arms = write_arms(rows)
    fb = D.FakeBackend()
    sm, defn, _ = P.build_baseline("rv2_U_t", arms["U"]["path"], seeds="0,1,2,3,4", testing=True, backend=fb,
                                   quiet=True)
    rb = sm.get("recovery_build") or {}
    check("the U arm named in arms.json builds with 5 seeds and records recovery_build (named by arm U, overlay "
          "rows matched, unmasked originals cleared)",
          defn["seeds"] == [0, 1, 2, 3, 4] and fb.submissions[0]["n"] == 5
          and rb.get("named_by", {}).get("arm") == "U" and rb["overlay_rows"] == len(rows) - 1
          and rb["matched_rows"] == rb["recovered_rows"] == len(rows) - 1 and not rb["problems"]
          and rb["unmasked_guard"]["hits"] == 0 and rb["unmasked_guard"]["checked"] == len(rows) - 1
          and rb["recovery"]["status"] == "complete"
          and json.loads((D.Paths("rv2_U_t").root / P.BUILD_SUMMARY).read_text())["recovery_build"] == rb, rb)
    sm, _d, _ = P.build_baseline("rv2_Uctl_t", arms["U_ctl"]["path"], testing=True, init=False, quiet=True)
    rb = sm.get("recovery_build") or {}
    check("the U_ctl arm (join labels on unmasked images, under step1_r1) is named in arms.json and builds",
          rb.get("named_by", {}).get("arm") == "U_ctl" and rb["overlay_rows"] == 0 and not rb["problems"]
          and rb["unmasked_guard"]["checked"] == len(rows) - 1, rb)
    other = TMP / "rv2_manifests" / "some_recovered.jsonl"
    C.write_manifest(other, arm_rows(rows)[:230])
    sm, _d, _ = P.build_baseline("rv2_other_p", other, init=False, quiet=True)
    rb = sm.get("recovery_build") or {}
    check("a production build of a manifest no arm names, whose recovered rows are recovered_pool rows, builds and "
          "records them", rb.get("named_by") is None and rb["recovered_rows"] == 10 and rb["matched_rows"] == 10
          and not rb["problems"], rb)
    sm, _d, _ = P.build_baseline("rv2_base_t", W0.base_b(), testing=True, init=False, quiet=True)
    check("base B's baseline summary has no recovery_build key (a non-recovery build is unchanged)",
          "recovery_build" not in sm and "recovery_build" not in json.loads(
              (D.Paths("rv2_base_t").root / P.BUILD_SUMMARY).read_text()))

    # one overlay label edited, the manifest re-stamped: no longer the recovered pool's row
    edited = TMP / "rv2_manifests" / "edited.jsonl"
    er = arm_rows(rows)[:230]
    tgt = next(r for r in er if "labels_overlay" in r["label"])
    saved = pathlib.Path(tgt["label"]).read_bytes()
    try:
        _append(tgt["label"])
        tgt["label_sha256"] = C.sha256_file(tgt["label"])
        C.write_manifest(edited, er)
        check("a manifest with one overlay label edited (re-stamped) refuses in production",
              raises(lambda: P.build_baseline("rv2_edit_p", edited, init=False, quiet=True), P.PilotError,
                     "not recovered_pool.jsonl rows") and W0.not_built("rv2_edit_p"))
        sm, _d, _ = P.build_baseline("rv2_edit_t", edited, testing=True, init=False, quiet=True)
        check("... and is recorded as a problem in a testing build",
              any("not recovered_pool.jsonl rows" in p for p in sm["recovery_build"]["problems"]))
    finally:
        pathlib.Path(tgt["label"]).write_bytes(saved)

    # the unmasked original of a masked copy near an evaluation image
    with W0.Edited():
        never = json.loads(C.NEVER_TRAIN_INDEX.read_text())["entries"]
        set_dhash({"u3_002": int(never[1][0]) ^ 0b100})
        _rec, rows2 = write_overlay(testing=False)
        near = TMP / "rv2_manifests" / "near.jsonl"
        base_part = C.read_manifest(W0.base_b())
        C.write_manifest(near, base_part + [{k: r[k] for k in C.MANIFEST_KEYS} for r in rows2
                                            if r["key"] in ("u3_002", "u3_003", "u1_009")])
        check("a production build of a manifest whose masked copy's unmasked original is within 6 bits of an "
              "evaluation image refuses (the masked copy itself clears check_training_manifest)",
              raises(lambda: P.build_baseline("rv2_near_p", near, init=False, quiet=True), P.PilotError,
                     "unmasked originals") and W0.not_built("rv2_near_p")
              and any(r["key"] == "u3_002" for r in C.read_manifest(near)))
    # arms.json vouches for an arm's control rows only while it is current and not a testing run
    _rec, rows = write_overlay(testing=False)
    arms = write_arms(rows, testing=True)
    check("a production build of an arm named by a testing arms.json refuses",
          raises(lambda: P.build_baseline("rv2_Uctl_tp", arms["U_ctl"]["path"], init=False, quiet=True),
                 P.PilotError, "testing recover --arms") and W0.not_built("rv2_Uctl_tp"))

    def stale(doc):
        doc["inputs"]["base"]["sha256"] = "0" * 64
    arms = write_arms(rows, mutate=stale)
    check("a production build of an arm whose arms.json inputs changed since (the base here) refuses",
          raises(lambda: P.build_baseline("rv2_Uctl_sp", arms["U_ctl"]["path"], init=False, quiet=True),
                 P.PilotError, "changed after it was written") and W0.not_built("rv2_Uctl_sp"))
    arms = write_arms(rows, mutate=lambda d: {k: v for k, v in d.items() if k != "inputs"})
    check("... and one whose arms.json records no inputs refuses",
          raises(lambda: P.build_baseline("rv2_Uctl_np", arms["U_ctl"]["path"], init=False, quiet=True),
                 P.PilotError, "records no inputs") and W0.not_built("rv2_Uctl_np"))
    arms = write_arms(rows)
    sm, _d, _ = P.build_baseline("rv2_Uctl_p", arms["U_ctl"]["path"], init=False, quiet=True)
    check("a production build of the U_ctl arm named by a current, non-testing arms.json builds",
          (sm.get("recovery_build") or {}).get("named_by", {}).get("arm") == "U_ctl"
          and not sm["recovery_build"]["problems"])
    with W0.Edited():
        other2 = TMP / "rv2_manifests" / "some_recovered2.jsonl"
        C.write_manifest(other2, arm_rows(rows)[:230])
        meta = read_meta()
        W0.write_rows(V.POOL_META, meta[::-1])          # an input recovery.json recorded, changed since
        check("a production build of a recovery manifest whose recovery.json inputs changed since refuses",
              raises(lambda: P.build_baseline("rv2_stale_p", other2, init=False, quiet=True), P.PilotError,
                     "changed after it was written") and W0.not_built("rv2_stale_p"))

    # recovery.json itself: a production build needs a complete, non-testing recovery
    for name, kw, contains in (("a testing recovery", {"testing": True}, "testing recover"),
                               ("an incomplete recovery", {"testing": False,
                                                           "mutate_rec": lambda r: dict(r, status="refused")},
                                "not 'complete'")):
        _rec, rows = write_overlay(**kw)
        man = TMP / "rv2_manifests" / "rec_state.jsonl"
        C.write_manifest(man, arm_rows(rows)[:230])
        exp = "rv2_recstate_%d" % len(contains)
        check("a production build of a recovery manifest of %s refuses" % name,
              raises(lambda: P.build_baseline(exp, man, init=False, quiet=True), P.PilotError, contains)
              and W0.not_built(exp))

    _rec, rows = write_overlay(testing=False)
    write_arms(rows)
    dd = C.read_manifest(R1 / "domain_dev.jsonl")
    pool = {r["key"]: r for r in C.read_manifest(V.POOL)}
    heldm = TMP / "rv2_manifests" / "held.jsonl"
    C.write_manifest(heldm, arm_rows(rows)[:225] + [pool[dd[0]["key"]]])
    check("a manifest holding an H10d domain-dev image refuses in any build, testing included",
          raises(lambda: P.build_baseline("rv2_held_t", heldm, testing=True, init=False, quiet=True), P.PilotError,
                 "domain dev") and W0.not_built("rv2_held_t"))


def main():
    W0.make_world()
    plant_groups()
    test_units()
    test_build()
    test_substitution()
    test_refusals()
    test_flags()
    test_baseline()


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), SKIPS))
    sys.exit(1 if FAILURES else 0)
