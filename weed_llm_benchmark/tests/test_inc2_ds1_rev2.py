#!/usr/bin/env python3
"""DS1 Revision 2, before any DS1 training or score (docs/CONTINUOUS_LOOP.md, "Revision (2026-10-29): DS1 Revision 2,
before any DS1 training or score"; docs/prereg/DS1_rev2_2026-10-29.md).

- item 1, the POWER pre-flight's weed check: on tied confidences (fp16, as on the V100) the key-order arrays and the
  validator-order AP differ (the refusal of jobs 50015220 and 50015542); a DS1 pass whose validator sees the images in
  another order than the exam's keys passes (ds1_pass end to end on a stubbed scorer), and its weed reading is the
  tie-broken one; weed_check refuses arrays that are not the stage-1 helper's records (a detection lost, a row changed,
  rows reordered within an image, another ground-truth count, an image recorded twice);
- item 3, the owner's audit render: the ticker records the set it rendered from its own proposal (not the tick's
  context); the plan proposes no render for a set the ticker rendered or the cluster lists rendered (the live state
  of 2026-10-29: [] recorded, audit_001 to audit_004 rendered for one set), and the pools step is proposed then;
  superseding: a rendered audit batch marked superseded (kept) is never recorded and no longer counts for its set;
  refusals (unknown sheet, another set, itself, twice, by a superseded sheet, one with decisions, no reason);
- item 4, the model audit: labelled boxes are read as (cx, cy, w, h) (a class-first box is refused); a screen's audit
  of a source whose labels a detector finds exactly is not partial; a revision audited before Revision 2 is taken
  again (reaudit: the same seeded sample, the old verdict kept beside), read in place of the old one by
  screen_records and by the sheets, before any pool (pools takes it first, refuses a pre-Revision-2 audit's masks);
  an 'accept' decision on a partially labelled source waits for a new decision; plan_state lists what is due;
- item 2, the power pre-flight recorded failed is reopened once the Revision is deployed (once, only for an attempt
  started before it) and proposed again;
- item 5, RA's default reports: the stream's segments since its fork only (never b_v2_s640 or a pre-fork segment);
- item 6 and the registration: Revision 2's byte copy hashes to its pin; a Revision made after the time-stamp record
  is named by its own record (timestamp_rev<n>.json), without which the gate stays closed; the time-stamp card is
  raised again for it, naming its step.

Run:  python3 tests/test_inc2_ds1_rev2.py
"""
import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
import traceback
import types

TMP = pathlib.Path(tempfile.mkdtemp(prefix="test_ds1_rev2_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
os.environ["INC_SCORER_TESTING"] = "1"
PKG = pathlib.Path(__file__).resolve().parents[1]
GIT = PKG.parent
sys.path.insert(0, str(PKG))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1 as D  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1_data as DD  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1_sheets as SH  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import diagnose_stream as DS  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import levers_stream as LS  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import stream as ST  # noqa: E402

FAILURES = []
DD.PROJECT_LOADER = False


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else "  %s" % (str(detail)[:900],)))
    if not cond:
        FAILURES.append(name)


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except (D.Ds1Refused, DD.DataRefused, SH.SheetRefused) as e:
        return e
    return None


@contextlib.contextmanager
def inc(name):
    prev = C.INC_DIR
    C.INC_DIR = TMP / name
    C.INC_DIR.mkdir(parents=True, exist_ok=True)
    try:
        yield C.INC_DIR
    finally:
        C.INC_DIR = prev


@contextlib.contextmanager
def patched(*items):
    """(object, attribute, value) set for the block, restored after."""
    saved = [(o, n, getattr(o, n)) for o, n, _v in items]
    for o, n, v in items:
        setattr(o, n, v)
    try:
        yield
    finally:
        for o, n, v in reversed(saved):
            setattr(o, n, v)


def jl(path, rows):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# ------------------------------------------------------------------ item 1: the weed check
def tied_world(seed=11, n_img=240, ties=True):
    """Per-image stage-1 records with fp16-like confidences (multiples of 2**-11, most near the top: E1-A seed 0 on dev
    holds 766 distinct values for 1,693 boxes) and the validator's image order (a shuffle of the exam's keys, as its
    aspect-ratio batches order them)."""
    rng = np.random.default_rng(seed)
    keys = ["img_%04d" % i for i in range(n_img)]
    capture = list(keys)
    rng.shuffle(capture)
    recs = {}
    for k in keys:
        n_gt = int(rng.integers(1, 5))
        n_p = int(rng.integers(0, 7))
        q = np.round(rng.beta(12.0, 1.0, n_p) * 2048.0) / 2048.0 if ties else rng.random(n_p)
        tp = np.zeros((n_p, 10), dtype=bool)
        hits = 0
        for j in np.argsort(-q, kind="stable"):
            if hits < n_gt and rng.random() < q[j]:
                tp[j, :int(rng.integers(1, 11))] = True
                hits += 1
        recs[k] = (tp, q.astype(np.float64), n_gt)
    return keys, capture, recs


def fake_v(capture, recs):
    v = types.SimpleNamespace(e3_s1_tp=[], e3_s1_q=[], e3_s1_ngt=0, d_weed={}, d_order=[])
    for k in capture:
        tp, q, ng = recs[k]
        v.e3_s1_tp.append(tp)
        v.e3_s1_q.append(q)
        v.e3_s1_ngt += ng
        v.d_weed[k] = {"tp": tp, "q": q, "n_gt": ng}
        v.d_order.append(k)
    return v


def capture_ap(v):
    return S.collapsed_ap(np.concatenate(v.e3_s1_tp, 0), np.concatenate(v.e3_s1_q), v.e3_s1_ngt)


def stub_pass(keys, capture, recs, mode="identity"):
    """inc2.ds1.ds1_pass on a stubbed scorer: E3's and V3F's layers replaced by a base that keeps the stage-1 records
    (E3's helper's lists), the scorer's pass feeding the images in the validator's order; DS1's own layer and every
    check of ds1_pass run as they are."""
    class FakeBase(object):
        def inc_reset(self):
            self.e3_s1_tp, self.e3_s1_q, self.e3_s1_ngt = [], [], 0
            self.f_seen, self.sc_images, self.e3_stats, self.f_stats, self.f_tags = [], {}, {}, {}, {}
            self.e3_gt_maxdiff, self.e3_gt, self.f_rows = 0.0, [], {}

        def e3_stage1(self, pred, pb):
            self.e3_s1_tp.append(pred["tp"])
            self.e3_s1_q.append(pred["q"])
            self.e3_s1_ngt += len(pb["cls"])
            return {"key": pb["key"]}

    def fake_score(weights, exam, out, **kw):
        v = S._VALIDATOR()
        v.inc_reset()
        for k in capture:
            tp, q, ng = recs[k]
            v.e3_stage1({"tp": tp, "q": q}, {"key": k, "cls": [12] * ng})
            v.f_seen.append(k)
        D.SC._CAPTURE.append(v)
        return {"key_order_sha256": C.sha256_text("\n".join(keys)), "per_class": {}, "map50_95": 0.0}

    with patched((D.F, "family_validator_class", lambda cfg: (FakeBase, FakeBase)), (D.S, "score", fake_score),
                 (D.SC, "check_capture", lambda pc, sc: (0.0, 0.0)), (D.SC, "flatten", lambda sc, ks: {}),
                 (D.C, "read_manifest", lambda p: [{"key": k} for k in keys]), (D.F, "_flat_rows", lambda *a: {})):
        return D.ds1_pass("w.pt", "dev", {"mode": mode}, "POWER")


def t_weed_check():
    print("item 1: the weed arrays against the stage-1 agnostic AP (POWER jobs 50015220 and 50015542)")
    keys, capture, recs = tied_world()
    v = fake_v(capture, recs)
    s1 = capture_ap(v)
    weed = D.weed_arrays(v, keys)
    key_ap = S.collapsed_ap(weed["tp"], weed["conf"], int(weed["n_gt"].sum()))
    q = np.concatenate([recs[k][1] for k in keys])
    check("the fixture has the live failure's shape: tied confidences (%d values for %d boxes), and the key-order "
          "arrays' AP differs from the validator-order AP by %.2e (> 1e-9, the check main applied)" % (
              len(np.unique(q)), len(q), abs(key_ap[0] - s1[0])), abs(key_ap[0] - s1[0]) > 1e-9 and
          len(np.unique(q)) < 0.6 * len(q), (key_ap, s1))
    # end to end first (it runs on 8ba5f75 too, where it is refused): DS1's pass with its own layer, on a scorer feeding the images in the validator's order
    try:
        ps = stub_pass(keys, capture, recs)
        err = None
    except D.Ds1Refused as e:
        ps, err = None, e
    check("a DS1 pass whose validator sees tied images in another order than the exam's keys is not refused, and "
          "records the check (the POWER pre-flight's refusal)", err is None and ps["stats"].get(
              "weed_recompute_capture_order") == 0.0 and abs(ps["stage1_agnostic"][0] - s1[0]) < 1e-15, err)
    got = refused(D.weed_check, v, keys, weed, s1)
    rec = D.weed_check(v, keys, weed, s1) if got is None else {}
    check("weed_check passes: in the validator's order the arrays give the AP exactly; the tie-broken key-order AP is "
          "within %.2f" % D.WEED_TIE_TOL, got is None and rec["weed_recompute_capture_order"] <= 1e-12 and
          rec["weed_tie_broken_vs_stage1"] <= D.WEED_TIE_TOL, (got, rec))
    tb = D.weed_values(weed)
    check("  the weed reading is the tie-broken AP (inc2.v3fam.statistic_value's 'agnostic'), the same under any "
          "row order of the image's records", abs(tb[0] - D.F.statistic_value(weed, "agnostic")) < 1e-15)
    # what weed_check refuses
    k0 = next(k for k in capture if len(recs[k][1]) >= 2 and len(set(recs[k][1].tolist())) == len(recs[k][1]))
    lost = dict(v.d_weed)
    lost[k0] = {"tp": recs[k0][0][1:], "q": recs[k0][1][1:], "n_gt": recs[k0][2]}
    e_lost = refused(D.weed_check, v, keys, D.weed_arrays(types.SimpleNamespace(d_weed=lost), keys), s1)
    flip = dict(v.d_weed)
    t2 = recs[k0][0].copy()
    t2[0, 0] = not t2[0, 0]
    flip[k0] = {"tp": t2, "q": recs[k0][1], "n_gt": recs[k0][2]}
    e_flip = refused(D.weed_check, v, keys, D.weed_arrays(types.SimpleNamespace(d_weed=flip), keys), s1)
    # rows reordered within an image of an exam without ties: the AP cannot tell, the row-for-row check does
    keys_u, cap_u, recs_u = tied_world(seed=5, n_img=60, ties=False)
    v_u = fake_v(cap_u, recs_u)
    k_u = next(k for k in cap_u if len(recs_u[k][1]) >= 2)
    rord = dict(v_u.d_weed)
    rord[k_u] = {"tp": recs_u[k_u][0][::-1], "q": recs_u[k_u][1][::-1], "n_gt": recs_u[k_u][2]}
    w_r = D.weed_arrays(types.SimpleNamespace(d_weed=rord), keys_u)
    s1_u = capture_ap(v_u)
    same_ap = abs(S.collapsed_ap(w_r["tp"], w_r["conf"], int(w_r["n_gt"].sum()))[0] - s1_u[0]) < 1e-15
    e_rord = refused(D.weed_check, v_u, keys_u, w_r, s1_u) if same_ap else None
    gt = dict(v.d_weed)
    gt[k0] = dict(gt[k0], n_gt=recs[k0][2] + 1)
    e_gt = refused(D.weed_check, v, keys, D.weed_arrays(types.SimpleNamespace(d_weed=gt), keys), s1)
    dup = types.SimpleNamespace(**dict(vars(v), d_order=v.d_order[:-1] + [v.d_order[0]]))
    e_dup = refused(D.weed_check, dup, keys, weed, s1)
    check("refused: a detection lost, a match changed, rows reordered within an image (same AP), one ground-truth box "
          "more, an image recorded twice in the validator's order",
          all(e is not None for e in (e_lost, e_flip, e_rord, e_gt, e_dup)),
          (e_lost, e_flip, e_rord, e_gt, e_dup))
    if ps is not None:
        check("  its weed arrays are the key-order ones, their reading tie-broken (score_doc's weed_map50_95)",
              [str(k) for k in ps["weed"]["keys"]] == keys and
              abs(D.weed_values(ps["weed"])[0] - rec.get("weed_tie_broken_map50_95", -1)) < 1e-15 and
              abs(ps["stats"]["weed_tie_broken_map50_95"] - D.weed_values(ps["weed"])[0]) < 1e-15)
    same = list(keys)
    ps2 = stub_pass(keys, same, recs)
    check("  a pass in key order gives the same weed arrays and reading", ps2 is not None and
          np.array_equal(ps2["weed"]["conf"], ps["weed"]["conf"]) if ps is not None else False)


# ------------------------------------------------------------------ item 3: the owner's audit render
def fake_run(st=None):
    """A StreamRun's DS1 bookkeeping without a ticker: its state, domain and no evidence (as when a job's end is read
    before the tick's context holds DS1's status)."""
    r = types.SimpleNamespace(st=st if st is not None else {}, dom=LS.load_domain("weed"), ev=None,
                              utc="2026-10-29T19:17:04Z")
    for name in ("_ds1", "_ds1_item"):
        setattr(r, name, types.MethodType(getattr(ST.StreamRun, name), r))
    r._ds1_key = ST.StreamRun._ds1_key
    return r


def _fs(pend, rendered, audit_sets=None, render_for=None, sources=None):
    st = {"phases": {"1": {"revisions": 1, "new_batches": 0, "sources": sources or {}}},
          "md_s0": {"drafts": 1, "draft": "draft_r001", "power": False, "frozen": False}, "arms": {},
          "sheets": {"source_batches": 0, "ln_batches": 0, "mtw_batches": 0, "rendered": rendered,
                     "audit_pending": pend, "audit_sets": audit_sets or {}}, "scores": {}, "verdicts": {},
          "long": {"arms": {}, "pair": None}}
    items = {"sheets:audit:%d" % i: "done" for i in range(1, 1 + len([x for x in rendered if
                                                                      x.startswith("audit_")]))}
    items["data:1:screen"] = "missing"
    items["score:POWER:draft_r001"] = "failed"          # as live: the pre-flight recorded failed (Revision 2 items 1-2)
    return {"status": st, "items": items, "today": "2026-10-29", "phase1_gb": 2.0, "audit_render_for": render_for}


def t_audit_render():
    print("item 3: the owner's audit batch rendered once per set")
    cfg = LS.load_domain("weed")["ds1"]
    run = fake_run()
    it = {"lever": "L31", "proposal": {"id": "p1", "params": {"kind": "audit", "batch": 2, "lab_inc": "/x",
                                                             "sources": "src_b,src_a"}, "ds1_phase": 1}}
    run._ds1_item(it)[1].update(state="running")
    ST.StreamRun._ds1_done(run, it)
    check("a finished audit render records the set its own proposal named (no evidence at hand, as live)",
          run.st["ds1"].get("audit_render_for") == ["src_a", "src_b"], run.st["ds1"].get("audit_render_for"))
    pend = ["src_a", "src_b"]
    _o, _w, c = DS.ds1_plan(cfg, _fs(pend, ["audit_001"], render_for=["src_a", "src_b"]))
    check("the plan proposes no render of a set the ticker rendered", not [x for x in c["maint"] if
                                                                        x["item"].startswith("sheets:audit")],
          c["maint"])
    live = _fs(pend, ["audit_001", "audit_002", "audit_003", "audit_004", "source_001"],
               audit_sets={"audit_00%d" % i: pend for i in range(1, 5)}, render_for=[],
               sources={"src_c": {"decided": True, "decision": "accept_training_only", "ready": True,
                                  "pooled": False}})
    _o, _w, c2 = DS.ds1_plan(cfg, live)
    check("the live state of 2026-10-29 ([] recorded, audit_001-004 rendered for one set): no fifth render, and the "
          "pools step is proposed on MAINT", not [x for x in c2["maint"] if x["item"].startswith("sheets:audit")] and
          [x["item"] for x in c2["maint"]][:1] == ["data:1:pools"], [x["item"] for x in c2["maint"]])
    _o, _w, c3 = DS.ds1_plan(cfg, _fs(["src_a", "src_b", "src_c"], ["audit_001"], audit_sets={"audit_001": pend},
                                      render_for=pend))
    au = [x for x in c3["maint"] if x["item"].startswith("sheets:audit")]
    check("  a new pending set is rendered as the next batch", au and au[0]["batch"] == 2 and au[0]["sources"] ==
          ["src_a", "src_b", "src_c"], au)
    _o, _w, c4 = DS.ds1_plan(cfg, _fs(pend, ["audit_001", "audit_002"], audit_sets={}, render_for=None))
    au4 = [x for x in c4["maint"] if x["item"].startswith("sheets:audit")]
    check("  a set whose every render is superseded (none listed) is rendered again", au4 and au4[0]["batch"] == 3,
          au4)
    check("  ds1_audit_sources reads a comma list or a list", ST.ds1_audit_sources(" b, a,") == ["a", "b"] and
          ST.ds1_audit_sources(["b", "a"]) == ["a", "b"] and ST.ds1_audit_sources(None) == [])


def render_rec(d, kind, batch, sources, spec):
    p = SH.rendered_path(kind, batch)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"format": SH.FMT_RENDER, "kind": kind, "batch": batch, "spec_sha256": spec,
                             "html": "x.html", "html_sha256": spec[::-1], "bundles": [], "sources": sources,
                             "items": [], "item_boxes": {}, "item_groups": {}, "item_totals": {},
                             "rendered_utc": "2026-10-29T18:37:09Z"}))


def decisions_file(d, spec, sources, name):
    p = d / name
    p.write_text(json.dumps({"format": SH.FMT_DECISIONS, "sheet": {"spec_sha256": spec}, "decider": "Harry",
                             "saved_utc": "2026-10-30T01:00:00Z", "sources": {
                                 s: {"decision": "reject", "setting": "Field", "roles": {}} for s in sources}}))
    return p


def t_supersede():
    print("item 3: an extra render marked superseded (kept, never recorded)")
    with inc("inc_sup") as d:
        srcs = ["src_a", "src_b"]
        for b in (1, 2, 3):
            render_rec(d, "audit", b, srcs, ("%d" % b) * 64)
        render_rec(d, "audit", 4, ["src_c"], "4" * 64)
        render_rec(d, "source", 1, srcs, "5" * 64)
        check("before: three audit batches of one set listed", sorted(SH.audit_sets()) == ["audit_001", "audit_002",
                                                                                          "audit_003", "audit_004"])
        e_unknown = refused(SH.supersede, "audit_009", "audit_001", "x")
        e_other = refused(SH.supersede, "audit_004", "audit_001", "x")
        e_kind = refused(SH.supersede, "source_001", "audit_001", "x")
        e_self = refused(SH.supersede, "audit_002", "audit_002", "x")
        e_why = refused(SH.supersede, "audit_002", "audit_001", " ")
        row = SH.supersede("audit_002", "audit_001", "rendered again for the same set (2026-10-29)")
        e_twice = refused(SH.supersede, "audit_002", "audit_001", "x")
        e_by = refused(SH.supersede, "audit_003", "audit_002", "x")
        check("supersede refuses an unknown sheet, another set, another kind, itself, no reason, twice, and a "
              "superseded sheet as the one that stands",
              all(e is not None for e in (e_unknown, e_other, e_kind, e_self, e_why, e_twice, e_by)),
              (e_unknown, e_other, e_kind, e_self, e_why, e_twice, e_by))
        check("  the row names the sheet, its sha256s, the sheet that stands and the reason; the files stay",
              row["sheet"] == "audit_002" and row["by"] == "audit_001" and row["spec_sha256"] == "2" * 64 and
              SH.rendered_path("audit", 2).is_file())
        st = SH.status()
        check("  status lists it superseded; the audit sets no longer count it", st["superseded"] == ["audit_002"] and
              sorted(st["audit_sets"]) == ["audit_001", "audit_003", "audit_004"], st)
        e_rec = refused(SH.record, str(decisions_file(d, "2" * 64, srcs, "dec2.json")), SH.OWNER_HUMAN)
        check("record refuses a decisions file naming the superseded sheet", e_rec is not None and "superseded" in
              str(e_rec), e_rec)
        SH.record(str(decisions_file(d, "3" * 64, srcs, "dec3.json")), SH.OWNER_HUMAN)
        e_dec = refused(SH.supersede, "audit_003", "audit_001", "x")
        check("  a sheet with recorded decisions is not superseded", e_dec is not None, e_dec)
        ps = DD.plan_state(d)
        check("plan_state carries the audit sets and the superseded sheets for the platform's plan",
              ps["sheets"]["superseded"] == ["audit_002"] and "audit_002" not in ps["sheets"]["audit_sets"] and
              ps["sheets"]["audit_sets"]["audit_001"] == srcs, ps["sheets"])


# ------------------------------------------------------------------ item 4: the model audit
def perfect(rows):
    """A detector that finds exactly each row's labelled boxes (conf 0.9), keyed by the row's key."""
    box = {r["key"]: [tuple(float(x) for x in b[1:5]) for b in r["boxes"]] for r in rows}
    return lambda r: [b + (0.9,) for b in box[r["key"]]]


def png(path, seed, size=(400, 400)):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)
    return str(path)


def h64(seed):
    return int(np.random.default_rng(seed).integers(0, 2 ** 63)) | (1 << 63)


HASHES = {}


def cand(d, key, source, seed, boxes):
    img = png(d / "src" / source / ("%s.png" % key), seed)
    hh = h64(seed)
    HASHES[img] = (hh, {k: h64(seed * 100 + i) if i else hh for i, k in enumerate(DD.B3.VARIANTS)})
    lab = d / "src" / source / ("%s.txt" % key)
    lab.write_text("".join("%d %.6f %.6f %.6f %.6f\n" % tuple(b) for b in boxes))
    return {"key": key, "source": source, "pool": "p3", "image": img, "sha256": C.sha256_file(img), "label": str(lab),
            "boxes": [tuple(b) for b in boxes], "W": 400, "H": 400, "rel": key, "series": None, "eppo": None,
            "capture": None, "stem": None, "session": "", "dhash": hh, "variants": None, "licence": "cc-by",
            "research_only": False, "holds": [], "refused": None, "admission": "whole", "test_v1": None,
            "batch": "b0012", "guard_match": None, "pair_cos": None, "box_src": None}


def screen_inputs(rows, predict=None, ref=None):
    return DD.Inputs(candidates=rows, scanners={}, intake_hits={}, record={"step1_batches": ["b0012"]},
                     exams={"dev": {"keys": ["dev__a"], "H": np.asarray([[h64(9002)] * 8], dtype=np.uint64),
                                    "ok": np.asarray([True])}},
                     hasher=lambda p: HASHES.get(str(p), (None, None)), production=False, predict=predict,
                     audit_ref=ref)


REF = {"images": 200, "predictions": 300, "gt_boxes": 320, "missing_rate": 0.021, "false_box_rate": 0.05,
       "convention_log_area": 0.0}


def t_audit():
    print("item 4: the model audit reads (cx, cy, w, h)")
    rows = [{"key": "r%d" % i, "boxes": [(0.3, 0.4, 0.2, 0.2)]} for i in range(3)]
    e = refused(DD.audit_rates, [{"key": "r0", "boxes": [(12, 0.3, 0.4, 0.2, 0.2)]}], lambda r: [])
    check("audit_rates refuses a class-first box (the screen's call before Revision 2)", e is not None, e)
    got = DD.audit_rates(rows, lambda r: [(0.3, 0.4, 0.2, 0.2, 0.9)])
    check("  boxes found exactly: missing 0, false boxes 0", got["missing_rate"] == 0.0 and
          got["false_box_rate"] == 0.0, got)
    with inc("inc_audit") as d:
        rs = [cand(d, "a_%d" % i, "src_full", 100 + i, [(12, 0.3, 0.4, 0.2, 0.2), (12, 0.7, 0.6, 0.15, 0.2)])
              for i in range(6)]
        rs += [cand(d, "p_%d" % i, "src_part", 200 + i, [(12, 0.3, 0.4, 0.2, 0.2)]) for i in range(6)]
        pred = perfect(rs)
        partial_pred = lambda r: pred(r) + ([(0.7, 0.7, 0.1, 0.1, 0.9), (0.1, 0.1, 0.1, 0.1, 0.8)]
                                            if r["source"] == "src_part" else [])
        doc = DD.screen(1, screen_inputs(rs, partial_pred, REF))
        a_full = doc["sources"]["src_full"]["audit_verdict"]
        a_part = doc["sources"]["src_part"]["audit_verdict"]
        check("a screen's audit of a source whose labels the detector finds exactly is not partial (missing 0, false "
              "boxes 0, version 2)", a_full and not a_full["partial"] and a_full["source"]["missing_rate"] == 0.0 and
              a_full["source"]["false_box_rate"] == 0.0 and a_full.get("version") == getattr(DD, "AUDIT_VERSION", 2),
              a_full)
        check("  one with confident unlabelled plants is (missing 2/3 > train_core's + 0.15)", a_part["partial"] and
              abs(a_part["source"]["missing_rate"] - 2 / 3.0) < 1e-12, a_part)
        check("  nothing to take again", DD.audit_due(1) == [])
    check("audit_boxes strips the class", DD.audit_boxes({"boxes": [(12, 0.1, 0.2, 0.3, 0.4)]}) ==
          [(0.1, 0.2, 0.3, 0.4)])
    with inc("inc_reaudit") as d:
        rs = [cand(d, "a_%d" % i, "src_full", 300 + i, [(12, 0.3, 0.4, 0.2, 0.2)]) for i in range(5)]
        rs += [cand(d, "m_%d" % i, "src_mask", 400 + i, [(12, 0.3, 0.4, 0.2, 0.2)]) for i in range(5)]
        pred = perfect(rs)
        with patched((DD, "source_audit", lambda rows, s, p, ref: {
                "partial": True, "excess": 0.979, "threshold": 0.15, "train_core": ref, "source": {
                    "images": len(rows), "predictions": 5, "gt_boxes": 5, "missing_rate": 1.0,
                    "false_box_rate": 1.0, "convention_log_area": None}})):
            DD.screen(1, screen_inputs(rs, pred, REF))
        rev = DD.screen_revisions(1)[0]
        old = DD._read_json(rev)["sources"]["src_full"]["audit_verdict"]
        check("a revision audited before Revision 2 (no version, partial everywhere) is due", DD.audit_due(1) == [rev]
              and old["partial"] and "version" not in old and DD.plan_state(d)["phases"]["1"]["audit_due"] ==
              [rev.name], DD.audit_due(1))
        nopred = DD.Inputs(decisions={"src_full": {"decision": "accept_training_only", "roles": {"inc:12": "weed"}}})
        e_np = refused(DD.pools, 1, nopred)
        check("  pools refuses while it is due and no predictor can take it again", e_np is not None, e_np)
        with patched((DD, "reaudit", lambda phase, inputs, write=True: [])):
            e_mask = refused(DD.pools, 1, DD.Inputs(decisions={"src_full": {
                "decision": "accept_training_only", "roles": {"inc:12": "weed"}}}, predict=pred, audit_ref=REF),
                write=False)
        check("  pools never applies masks from an audit taken before Revision 2", e_mask is not None and
              "taken before Revision 2" in str(e_mask), e_mask)
        inputs = screen_inputs(rs, pred, REF)
        pred.weights_sha256 = "0" * 64
        e_w = refused(DD.reaudit, 1, inputs, write=False)
        pred.weights_sha256 = DD.AUDIT_PREDICTOR_SHA256
        check("  reaudit refuses another predictor than the screen's (E1-A seed 0's weights by sha256)",
              e_w is not None and "E1-A seed 0" in str(e_w) and refused(DD.reaudit, 1, inputs, write=False) is None,
              e_w)
        later = types.SimpleNamespace(**vars(inputs))
        later.never = {"src_mask"}                  # a never-train entry made after the screen
        gone = DD.reaudit(1, later, write=False)[0]
        check("  a source whose live rows are no longer the screen's is on record (rows_as_screened, rows_differ)",
              gone["rows_differ"] == ["src_mask"] and gone["sources"]["src_mask"]["rows_as_screened"] is False and
              gone["sources"]["src_mask"]["live_now"] == 0 and gone["sources"]["src_full"]["rows_as_screened"],
              {k: gone[k] for k in ("rows_differ",)})
        done = DD.reaudit(1, inputs)
        ra = DD._read_json(DD.reaudit_path(rev))
        new = ra["sources"]["src_full"]["audit_verdict"]
        check("reaudit takes it again on the same live rows and seeded sample: not partial, version 2, the old verdict "
              "kept beside, partial_changed", len(done) == 1 and not new["partial"] and new["version"] ==
              DD.AUDIT_VERSION and ra["sources"]["src_full"]["superseded"] == old and ra["sources"]["src_full"][
                  "partial_changed"] and new["source"]["images"] == 5, ra["sources"]["src_full"])
        check("  the record names the predictor, the rows as screened and each sample's keys", ra[
            "predictor_weights_sha256"] == DD.AUDIT_PREDICTOR_SHA256 and ra["rows_differ"] == [] and ra["sources"][
            "src_full"]["sample_keys"] == [r["key"] for r in DD.audit_sample([x for x in rs if x["source"] ==
                                                                               "src_full"], "src_full")],
              {k: ra.get(k) for k in ("predictor_weights_sha256", "rows_differ")})
        check("  written once, beside the revision and under the sheets (the lab's pull carries it)",
              refused(DD.reaudit, 1, inputs) is None and DD.reaudit_sheet_path(rev).is_file() and
              DD.audit_due(1) == [] and refused(DD.write_once, DD.reaudit_path(rev), {}) is not None)
        srec = DD.screen_records(1)
        check("screen_records reads the audit taken again, the old one beside it", srec["src_full"][
            "audit_verdict"] == new and srec["src_full"]["audit_superseded"] == old)
        av = SH.audit_v2()
        check("  the sheets read it per (phase, revision, source)", av[(1, 1, "src_full")]["audit_verdict"] == new)
        bname = "source_p1_r001_src_full"
        (SH.bundle_dir(bname) / "bundle.json").parent.mkdir(parents=True, exist_ok=True)
        (SH.bundle_dir(bname) / "bundle.json").write_text("{}")
        meta = {"phase": 1, "revision": 1, "source": "src_full", "screen": {"audit_verdict": old, "live": 5}}
        with patched((SH, "verify_bundle", lambda n, inc=None: {"items": [], "meta": meta})):
            spec, _f = SH.sheet_spec("source", 1, [bname], decisions={})
        sc = spec["sources"][0]["screen"]
        check("  a sheet rendered after it shows the audit taken again, the screen's beside it",
              sc["audit_verdict"] == new and sc["audit_superseded"] == old and sc["live"] == 5, sc)
        check("an 'accept' decision on a partially labelled source (version 2) waits for a new decision; for training "
              "only it is ready; an audit taken before Revision 2 holds nothing",
              DD.source_ready("s", {"s": {"decision": "accept"}}, {"audit_verdict": dict(new, partial=True)})[0]
              is False and DD.source_ready("s", {"s": {"decision": "accept_training_only"}},
                                           {"audit_verdict": dict(new, partial=True)})[0] is True and
              DD.source_ready("s", {"s": {"decision": "accept"}}, {"audit_verdict": old})[0] is True)
    with inc("inc_pools_first") as d:
        rs = [cand(d, "a_%d" % i, "src_full", 500 + i, [(12, 0.3, 0.4, 0.2, 0.2)]) for i in range(4)]
        pred = perfect(rs)
        with patched((DD, "source_audit", lambda rows, s, p, ref: {
                "partial": True, "excess": 0.979, "threshold": 0.15, "train_core": ref,
                "source": {"images": len(rows), "missing_rate": 1.0}})):
            DD.screen(1, screen_inputs(rs, pred, REF))
        seen = []
        real = DD.reaudit

        def spy(phase, inputs, write=True):
            seen.append(phase)
            return real(phase, inputs, write=write)
        with patched((DD, "reaudit", spy)):
            try:
                DD.pools(1, DD.Inputs(decisions={"src_full": {"decision": "accept_training_only",
                                                              "roles": {"inc:12": "weed"}}}, predict=pred,
                                      audit_ref=REF, candidates=rs, hasher=lambda p: HASHES.get(str(p))))
            except Exception:                       # noqa: BLE001 - the pool itself needs inputs this test omits
                pass
        check("pools takes the audit again first (before it reads any audit)", seen == [1] and DD.audit_due(1) == [],
              (seen, DD.audit_due(1)))


def t_sheet_label():
    print("item 4: the sheets show the audit taken again, and label one taken before Revision 2")
    spec = {"kind": "source", "batch": 1, "spec_sha256": "s" * 64, "rendered_utc": "t", "bundles": [], "items": [],
            "sources": [{"source": "src_x", "units": [], "screen": {"audit_verdict": {"partial": True}},
                         "unit_info": {}, "delegate": None, "items": []}]}
    h_old = SH.render_html(spec, {}).decode("utf-8") if isinstance(SH.render_html(spec, {}), bytes) else \
        SH.render_html(spec, {})
    spec["sources"][0]["screen"]["audit_verdict"] = {"partial": False, "version": DD.AUDIT_VERSION}
    h_new = SH.render_html(spec, {})
    h_new = h_new.decode("utf-8") if isinstance(h_new, bytes) else h_new
    check("an audit without a version is labelled not valid; one of version 2 as Revision 2's",
          "taken before Revision 2" in h_old and "model audit (Revision 2)" in h_new and
          "taken before Revision 2" not in h_new)


def t_reopen():
    print("item 2: the power pre-flight recorded failed is proposed again once Revision 2 is deployed")
    blk = {"marker": "DS1 Revision 2 (test)", "file": "docs/prereg/x.md", "sha256": "s" * 64,
           "items": ["score:POWER:draft_r001"], "before_utc": "2026-10-29T21:00:00Z", "why": "the weed check"}
    dom = LS.load_domain("weed")
    dom["ds1"] = dict(dom["ds1"], reopens=[blk])
    deployed = {"ok": True}
    st = {"failed_ids": ["other"], "ds1": {"items": {"score:POWER:draft_r001": {
        "state": "failed", "fails": 2, "proposal": "pidB", "spend": "settled", "settled": 0.03,
        "started_utc": "2026-10-29T18:17:04Z", "attempts": [{"proposal": "pidA", "gpuh": 0.05, "how": "settled"}],
        "prior_gpuh": 0.05, "why": "the weed arrays give AP50-95 0.856888007"}}, "cards": {}}}
    run = fake_run(st)
    run.dom = dom
    run.ledger, run.cards = [], []
    run._ds1_tree_file = lambda rel, sha=None, marker=None: deployed["ok"] and rel == blk["file"] and sha == blk["sha256"]
    run._failed_ids = types.MethodType(ST.StreamRun._failed_ids, run)
    run._v3fam_fold = ST.StreamRun._v3fam_fold
    run._ledger = lambda kind, **kw: run.ledger.append((kind, kw))
    run._card = lambda kind, title, body, **kw: run.cards.append(title)
    deployed["ok"] = False
    check("not reopened while the deployed tree lacks the Revision's file", ST.StreamRun._ds1_reopen(run) == [] and
          st["ds1"]["items"]["score:POWER:draft_r001"]["state"] == "failed")
    deployed["ok"] = True
    got = ST.StreamRun._ds1_reopen(run)
    rec = st["ds1"]["items"]["score:POWER:draft_r001"]
    check("reopened once deployed: state missing, fails 0, earlier ids failed (new ids), spend kept in prior_gpuh, "
          "history kept, one card and one ledger line", got == ["score:POWER:draft_r001"] and rec["state"] == "missing"
          and rec["fails"] == 0 and {"pidA", "pidB"} <= set(st["failed_ids"]) and abs(rec["prior_gpuh"] - 0.08) <
          1e-12 and rec["reopen_history"][-1]["marker"] == blk["marker"] and len(run.cards) == 1 and
          run.ledger[0][0] == "ds1_reopened", (got, rec, run.cards))
    check("  its card says reopened, not proposed (nothing is proposed while the gate is closed)",
          "reopened" in run.cards[0] and "proposed again" not in run.cards[0], run.cards)
    rec.update(state="failed", started_utc="2026-10-29T18:30:00Z")
    check("  never twice under one marker", ST.StreamRun._ds1_reopen(run) == [] and rec["state"] == "failed")
    st["ds1"]["items"]["score:POWER:draft_r001"] = {"state": "failed", "started_utc": "2026-10-30T02:00:00Z",
                                                    "proposal": "pidC"}
    check("  an attempt started after the Revision is not reopened by it", ST.StreamRun._ds1_reopen(run) == [])
    cfg = dom["ds1"]
    fs = _fs([], ["source_001"])
    fs["items"]["score:POWER:draft_r001"] = "missing"
    _o, _w, c = DS.ds1_plan(cfg, fs)
    check("  reopened, the plan proposes the power pre-flight (L33 POWER) again",
          [x["item"] for x in c["maint"]][:1] == ["score:POWER:draft_r001"], [x["item"] for x in c["maint"]])


def t_ts_card():
    print("item 6: the time-stamp card is raised again for a Revision made after the record, naming its own step")
    st = {"ds1": {"items": {}, "cards": {"timestamp": ST._sha("missing")}}}      # the live mark (raised before)
    run = fake_run(st)
    run.cards = []
    run._card = lambda kind, title, body, **kw: run.cards.append((title, body))
    run._ds1_card_once = types.MethodType(ST.StreamRun._ds1_card_once, run)
    status = {"timestamp": False, "timestamp_missing": {"record": True, "revisions": [2]}}
    ctx = {"stage": {"ds1": {"amendment": True, "never_cut": True, "status": status}}}
    run.ev = types.SimpleNamespace(json=lambda name: ctx)
    ST.StreamRun._ds1_cards(run, None)
    body = run.cards[0][1] if run.cards else ""
    check("the card asked for before is raised again, naming timestamp-revision --revision 2 with Revision 2's proofs",
          len(run.cards) == 1 and "timestamp-revision --revision 2 --proof docs/prereg/DS1_rev2_2026-10-29.md.ots "
          "--stamp-proof docs/prereg/DS1_rev2_2026-10-29.stamp.txt.ots" in body, run.cards)
    check("  only the Revision the status lists as unnamed, by its own number (never Revision 1, which the record "
          "names and timestamp-revision refuses), and not the record's own step; short enough for the ledger's copy",
          "--revision 1" not in body and "ds1 timestamp --proof" not in body and len(body) <= 1000, body)
    ST.StreamRun._ds1_cards(run, None)
    check("  and only once", len(run.cards) == 1)
    status["timestamp_missing"] = {"record": False, "revisions": [1, 2]}
    ST.StreamRun._ds1_cards(run, None)
    body = run.cards[-1][1] if len(run.cards) == 2 else ""
    check("  with no record yet: the record's own step with every Revision's proofs, no timestamp-revision",
          "ds1 timestamp --proof docs/prereg/DS1_2026-10-29.md.ots" in body and "--revision-proof docs/prereg/"
          "DS1_rev2_2026-10-29.md.ots" in body and "timestamp-revision" not in body, run.cards[1:])


def t_decide_again():
    print("item 4: an 'accept' on a partially labelled source waits for a new decision, named on a card")
    new = {"partial": True, "version": DD.AUDIT_VERSION}
    ok, why = DD.source_ready("s_acc", {"s_acc": {"decision": "accept"}}, {"audit_verdict": new})
    src = {"decided": True, "decision": "accept", "ready": ok, "why": why, "pooled": False}
    settled, waits = DS._ds1_sources_settled({"sources": {"s_acc": src}})
    check("the plan waits for its sheet (not for a pool)", not settled and waits == {"sheets": ["s_acc"]}, waits)
    run = fake_run({"ds1": {"items": {}, "cards": {}}})
    run.cards = []
    run._card = lambda kind, title, body, **kw: run.cards.append((title, body))
    run._ds1_card_once = types.MethodType(ST.StreamRun._ds1_card_once, run)
    ctx = {"stage": {"ds1": {"amendment": True, "never_cut": True, "status": {
        "timestamp": True, "ra": True, "phases": {"1": {"sources": {"s_acc": src}}}, "sheets": {}}}}}
    run.ev = types.SimpleNamespace(json=lambda name: ctx)
    run._artifact = lambda *a, **k: None
    ST.StreamRun._ds1_cards(run, {})
    check("  a card names it and the record step", any("a new sheet decision waits" in b and "s_acc" in b
                                                       for _t, b in run.cards), run.cards)


# ------------------------------------------------------------------ RA's default reports
def t_ra_reports():
    print("RA's default reports: the stream's segments since its fork")
    with inc("inc_ra") as d:
        for n in ("b_v2_s640", "weed_stream_v1_s001", "weed_stream_v1_s002", "weed_stream_v1_fork1364_s001",
                  "weed_stream_v1_fork1364_s002", "weed_stream_v1_fork1364_s003x", "other_fork1364_s001"):
            (d / n).mkdir(parents=True, exist_ok=True)
            (d / n / "report.json").write_text("{}")
        (d / "weed_stream_v1_fork1364_s004").mkdir()
        got = [pathlib.Path(p).parent.name for p in D.ra_reports()]
        check("only weed_stream_v1_fork1364_s<3 digits> with a report (never b_v2_s640 or a pre-fork segment)",
              got == ["weed_stream_v1_fork1364_s001", "weed_stream_v1_fork1364_s002"], got)


# ------------------------------------------------------------------ the registration: Revision 2
def t_registration():
    print("the registration: Revision 2 and its time stamp after the record")
    revs = {r["n"]: r for r in D.REVISIONS}
    check("DS1 has Revisions 1 and 2, in order", sorted(revs) == [1, 2] and [r["n"] for r in D.REVISIONS] == [1, 2])
    rv = revs.get(2) or {}
    f = GIT / rv.get("file", "missing")
    check("Revision 2's byte copy hashes to its pin; the domain's gate pins the same", f.is_file() and
          hashlib.sha256(f.read_bytes()).hexdigest() == rv.get("sha256") and
          [(r["file"], r["sha256"]) for r in LS.load_domain("weed")["ds1"]["revisions"]] ==
          [(r["file"], r["sha256"]) for r in D.REVISIONS])
    cl = (GIT / "docs" / "CONTINUOUS_LOOP.md").read_bytes()
    check("  and it is docs/CONTINUOUS_LOOP.md's Revision 2 section byte for byte (to the end of the file)",
          f.is_file() and cl.endswith(f.read_bytes()))
    with inc("inc_ts") as d:
        D.write_json(D.timestamp_path(), {"format": D.FMT_TIMESTAMP, "sha256": D.PREREG_SHA256, "proofs": [
            {"file": "a.ots", "sha256": "a" * 64}, {"file": "b.ots", "sha256": "b" * 64}],
            "revisions": [{"file": revs[1]["file"], "sha256": revs[1]["sha256"], "proofs": [
                {"file": "c", "sha256": "c" * 64}, {"file": "d", "sha256": "d" * 64}]}]})
        check("the live record (the amendment and Revision 1 only) does not open the gate once Revision 2 exists",
              not D.timestamp_ok() and refused(D.require_timestamp) is not None)
        check("  timestamp_missing: the record exists, Revision 2 is unnamed",
              D.timestamp_missing() == {"record": True, "revisions": [2]}, D.timestamp_missing())
        ots = [str(GIT / (rv["file"] + ".ots")), str(GIT / (rv["stamp_file"] + ".ots"))]
        e_sw = refused(D.timestamp_revision, 2, ots[1], ots[0])
        e_r1 = refused(D.timestamp_revision, 2, str(GIT / (revs[1]["file"] + ".ots")),
                       str(GIT / (revs[1]["stamp_file"] + ".ots")))
        e_n = refused(D.timestamp_revision, 3, ots[0], ots[1])
        e_named = refused(D.timestamp_revision, 1, str(GIT / (revs[1]["file"] + ".ots")),
                          str(GIT / (revs[1]["stamp_file"] + ".ots")))
        check("timestamp_revision refuses swapped proofs, Revision 1's proofs for Revision 2, an unknown Revision, and a "
              "Revision the record names already", all(e is not None for e in (e_sw, e_r1, e_n, e_named)),
              (e_sw, e_r1, e_n, e_named))
        rec = D.timestamp_revision(2, ots[0], ots[1])
        check("  with its committed proofs it writes timestamp_rev2.json (what each proof stamps, the record it "
              "follows), once; then the gate opens", rec["proofs"][0]["stamps"] == rv["sha256"] and
              rec["follows"]["sha256"] == C.sha256_file(D.timestamp_path()) and D.timestamp_ok() and
              refused(D.timestamp_revision, 2, ots[0], ots[1]) is not None)
        check("  timestamp_missing then lists nothing", D.timestamp_missing() == {"record": True, "revisions": []},
              D.timestamp_missing())
        bad = dict(json.loads(D.revision_stamp_path(2).read_text()), sha256="0" * 64)
        D.revision_stamp_path(2).write_text(json.dumps(bad))
        check("  a revision record naming another sha256 keeps it closed", not D.timestamp_ok() and
              D.timestamp_missing()["revisions"] == [2])
    with inc("inc_ts_none"):
        check("no record: timestamp_missing names the record and every Revision",
              D.timestamp_missing() == {"record": False, "revisions": [r["n"] for r in D.REVISIONS]},
              D.timestamp_missing())
    check("the domain's Revisions carry their own numbers (the card names each by it)",
          [r.get("n") for r in LS.load_domain("weed")["ds1"]["revisions"]] == [r["n"] for r in D.REVISIONS])


def main():
    for fn in (t_weed_check, t_reopen, t_audit_render, t_supersede, t_audit, t_sheet_label, t_ra_reports,
               t_registration, t_ts_card, t_decide_again):
        try:
            fn()
        except Exception as e:                   # noqa: BLE001
            check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-2500:]))
    print("\n%d failure(s), 0 skipped: %s" % (len(FAILURES), "; ".join(FAILURES[:5])))
    shutil.rmtree(str(TMP), ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
