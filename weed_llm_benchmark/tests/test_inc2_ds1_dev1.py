#!/usr/bin/env python3
"""DS1 deviation 1 (docs/CONTINUOUS_LOOP.md, "Deviation (2026-10-30): DS1 deviation 1"; docs/prereg/DS1_dev1_2026-10-30.md).

- item 1 and the live refusal: the power pre-flight's b1_ood23 reproduction refused on 2026-10-30 (jobs 50026958 and
  50027968): "V3F's recorded ood23 statistic of B'1 seed 0 is 0.149603, not the registered 0.2562". With V3F's live
  readings (P7's statistic 0.149603, 0.159426, 0.163401; the plain stage-1 agnostic AP 0.256235, 0.251712, 0.259808)
  and a pass reproducing them, the reproduction passes and records both readings per seed (refused on dd4c302);
- item 2, both readings named: each registered constant is checked against its own reading before any pass; the
  text's constants under P7's name refuse before the GPU runs (in the reopened pre-flight's world too, where e1a_dev
  and e3a_dev are due again: no record moves and no pass of any reproduction runs), the refusal naming both
  readings and saying the constant is the plain reading's; a wrong plain constant refuses too; each pass's two
  readings are held to DS.7.1's rule (a deviation within 0.002, refused beyond, whichever reading moved); P7's
  reading is the target arrays (t_), never the OtherPlant ones (o_);
- item 3, the reproduction records taken by the refused job under the earlier code (e1a_dev, e3a_dev) refuse every
  later score (require_reproductions); the pre-flight moves exactly those records aside, byte for byte, and takes them
  again, after which require_reproductions passes; a file with another sha256 stays; a move cut off after its link is
  completed; nothing is deleted;
- item 4, the reopen: the live state's score:POWER:draft_r001 (recorded failed 2026-10-30T03:22Z, reopened once under
  Revision 2) is reopened under the deviation's marker once the deployed tree holds its file, its spend kept in
  prior_gpuh (0.4571 GPU-h), its proposal id failed; never twice, never for an attempt after the deviation;
- the gate holds every DS1 lever until the deployed tree holds the deviation's file by its pinned sha256;
- the registration: the deviation's byte copy hashes to its pin (inc2.ds1.DEVIATIONS, the domain's ds1.deviations
  and its reopen block), ends docs/CONTINUOUS_LOOP.md, and its proofs stamp it and its stamp file, which names its
  sha256, its commit and the superseded records' sha256s.

Run:  python3 tests/test_inc2_ds1_dev1.py
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

TMP = pathlib.Path(tempfile.mkdtemp(prefix="test_ds1_dev1_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
os.environ["INC_SCORER_TESTING"] = "1"
PKG = pathlib.Path(__file__).resolve().parents[1]
GIT = PKG.parent
sys.path.insert(0, str(PKG))

import numpy as np  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1 as D  # noqa: E402
from weed_optimizer_framework.tools.inc2 import twostage as TS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_loader as LD  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_systems as SYS  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import diagnose_stream as DS  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import levers_stream as LS  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import stream as ST  # noqa: E402

FAILURES = []
# V3F's records of B'1 (v3fam/v1/arms/D1B1/s<k>/ood23.json, read on the cluster 2026-10-30): P7's statistic (V3F-R's P7
# x_values) and the plain stage-1 agnostic AP; the refused jobs' line
P7_LIVE = (0.1496034855232349, 0.1594256305243338, 0.16340065412424704)
PLAIN_LIVE = (0.2562354209154568, 0.251711982313296, 0.2598083587383618)
LIVE_REFUSAL = "V3F's recorded ood23 statistic of B'1 seed 0 is 0.149603, not the registered 0.2562"
POWER = "score:POWER:draft_r001"


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else "  %s" % (str(detail)[:900],)))
    if not cond:
        FAILURES.append(name)


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except D.Ds1Refused as e:
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


def arrays(seed, n_img=40):
    """Per-image agnostic arrays (keys, tp, conf, pred_img, n_gt) as a p7 sidecar holds them."""
    rng = np.random.default_rng(seed)
    keys, tp, conf, img, ngt = [], [], [], [], []
    for i in range(n_img):
        keys.append("ood23__img_%03d" % i)
        g = int(rng.integers(1, 4))
        ngt.append(g)
        for _j in range(int(rng.integers(0, 5))):
            hit = rng.random() < 0.6
            tp.append([hit and t < int(rng.integers(1, 11)) for t in range(10)])
            conf.append(float(rng.random()))
            img.append(i)
    return {"keys": np.asarray(keys), "tp": np.asarray(tp, dtype=bool).reshape(-1, 10),
            "conf": np.asarray(conf, dtype=np.float64), "pred_img": np.asarray(img, dtype=np.int64),
            "n_gt": np.asarray(ngt, dtype=np.int64)}


def v3f_b1(seed, plain, t=None, o=None):
    """V3F's D1B1 ood23 score file of one seed with its p7 sidecar (target and OtherPlant readings)."""
    d = D.F.arm_dir("D1B1", seed)
    d.mkdir(parents=True, exist_ok=True)
    t = t if t is not None else arrays(10 + seed)
    o = o if o is not None else arrays(20 + seed)
    flat = {"t_" + k: v for k, v in t.items()}
    flat.update({"o_" + k: v for k, v in o.items()})
    sha = D.SC.save_npz(d / "ood23.p7.npz", flat)
    doc = {"format": "inc2-v3fam-score/1", "arm": "D1", "build": "D1B1", "exam": "ood23", "seed": seed,
           "agnostic_map50_95": plain, "stage1": {"agnostic_map50_95": plain, "agnostic_map50": 0.4},
           "p7": {"name": "ood23.p7.npz", "sha256": sha}}
    (d / "ood23.json").write_text(json.dumps(doc))
    return d


def equiv(item, code=None, extra=None):
    doc = {"format": D.FMT_EQUIV, "item": item, "status": "passed", "per": {"0": {"diff": 0.0}},
           "code": code if code is not None else D.code_shas(), "created_utc": "2026-10-30T02:56:18Z"}
    doc.update(extra or {})
    p = D.equiv_dir() / ("%s.json" % item)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=1, sort_keys=True))
    return p


class Stubs(object):
    """The GPU side of reproductions(): run_source, the loader pre-flight, the deciding loader, DS1's pass (dev: E1-A's
    stage-1 and E3-A's species reading; ood23: B'1's two readings, each shifted as asked) and E3's classifier, with the
    calls counted. V3F's recorded P7 statistic is the live one (v3f_p7_value), V3F's plain reading the file's."""

    def __init__(self, dp7=0.0, dplain=0.0, p7=P7_LIVE, o_val=0.24):
        self.dp7, self.dplain, self.p7, self.o_val = dp7, dplain, p7, o_val
        self.passes, self.preflights, self.read_dirs = [], 0, []

    def run_source(self, exp, s, testing_ok=False, read_dev=True):
        return {"exp": exp, "seed": int(s), "weights": "%s_s%d.pt" % (exp, int(s))}

    def preflight(self, exams=("ood22", "ood23"), threads=None):
        self.preflights += 1
        return {}

    @contextlib.contextmanager
    def deciding(self, exam):
        yield None

    def ds1_pass(self, weights, exam, cfg, arm, **kw):
        self.passes.append((weights, exam))
        if exam == "dev":
            return {"stage1_agnostic": (0.8568578547397661, 0.95), "result": {"species_map50_95": 0.8859}}
        s = int(weights.rsplit("_s", 1)[1].split(".")[0])
        return {"p7": {"t": {"_v": self.p7[s] + self.dp7}, "o": {"_v": self.o_val}},
                "stage1_agnostic": (PLAIN_LIVE[s] + self.dplain, 0.41)}

    def v3f_p7_value(self, d, exam="ood23"):
        self.read_dirs.append(pathlib.Path(d))
        return self.p7[int(pathlib.Path(d).name[1:])]

    def items(self):
        orig = D.F.statistic_value

        def stat(a, statistic="species", *x, **k):
            return a["_v"] if isinstance(a, dict) and "_v" in a else orig(a, statistic, *x, **k)

        @contextlib.contextmanager
        def workers(w=None, **k):
            yield None

        return [(SYS, "run_source", self.run_source), (LD, "preflight", self.preflight), (LD, "deciding", self.deciding),
                (D, "ds1_pass", self.ds1_pass), (D, "v3f_p7_value", self.v3f_p7_value), (D.F, "statistic_value", stat),
                (TS, "_clf_cfg", lambda fold=None: ({}, None, {"classifier": "c" * 64})), (TS, "_workers", workers),
                (TS, "_source", lambda arm, s, testing_ok=False: {"weights": "e1a_s%d.pt" % int(s), "recorded": {
                    "agnostic_map50_95": 0.8568578547397661}}),
                (V, "BioclipEmbedder", lambda *a, **k: object()),
                (SYS, "_two_stage_cfg", lambda ccfg, letter, emb, wk, mode="two_stage": {"mode": mode, "embedder": emb})]


def b1_world():
    """V3F's three D1B1 ood23 files with the live plain readings, and every reproduction but b1_ood23 in place."""
    for s in D.SEEDS:
        v3f_b1(s, PLAIN_LIVE[s])
    for item in ("e1a_dev", "e3a_dev", "md_identity"):
        equiv(item)


# ------------------------------------------------------------------ item 1: the live refusal
def t_live():
    print("item 1: the power pre-flight's b1_ood23 on V3F's live readings (jobs 50026958 and 50027968)")
    with inc("inc_live"):
        b1_world()
        stb = Stubs()
        with patched(*stb.items()):
            err = refused(D.reproductions)
        check("with V3F's live readings and a pass reproducing them, b1_ood23 is not refused (dd4c302: '%s')"
              % LIVE_REFUSAL, err is None, err)
        rec = json.loads((D.equiv_dir() / "b1_ood23.json").read_text()) if err is None else {}
        per = rec.get("per") or {}
        check("  it passed: each seed's diff 0, P7's statistic the reading held (new, recorded)",
              rec.get("status") == "passed" and sorted(per) == ["0", "1", "2"] and
              all(per[str(s)]["diff"] == 0.0 and per[str(s)]["recorded"] == P7_LIVE[s] for s in D.SEEDS), rec)
        check("  each seed names both readings: P7's statistic and the plain one, recorded, new, diff and the "
              "registered constant", all(set(per[str(s)]["readings"]) == {"p7", "plain"} and
                                         per[str(s)]["readings"]["plain"]["recorded"] == PLAIN_LIVE[s] and
                                         per[str(s)]["readings"]["p7"]["registered_4dp"] == round(P7_LIVE[s], 4) and
                                         per[str(s)]["readings"]["plain"]["registered_4dp"] == round(PLAIN_LIVE[s], 4)
                                         for s in D.SEEDS) if per else False, per)
        check("  the record names the registered constants of both readings and the deviation's file",
              rec.get("registered") == {"p7": [0.1496, 0.1594, 0.1634], "plain": [0.2562, 0.2517, 0.2598]} and
              (rec.get("deviation") or {}).get("file") == "docs/prereg/DS1_dev1_2026-10-30.md", rec.get("registered"))
        check("  V3F's records were read from D1B1's own score directories, one GPU pass per seed",
              sorted({p.parent.name for p in stb.read_dirs}) == ["D1B1"] and
              [x for x in stb.passes if x[1] == "ood23"] == [("v3_d1_b1_m640_s%d.pt" % s, "ood23") for s in D.SEEDS],
              (stb.read_dirs, stb.passes))
    check("the registered constants are V3F P7's statistic, the plain reading's beside them (4 decimals)",
          D.B1_OOD23 == {"p7": tuple(round(x, 4) for x in P7_LIVE), "plain": tuple(round(x, 4) for x in PLAIN_LIVE)},
          D.B1_OOD23)


# ------------------------------------------------------------------ item 2: both readings named
def t_mislabel():
    print("item 2: each constant against its own reading, before any pass; the refusal names both readings")
    text = {"p7": tuple(round(x, 4) for x in PLAIN_LIVE), "plain": tuple(round(x, 4) for x in PLAIN_LIVE)}
    with inc("inc_mis1"):
        b1_world()
        stb = Stubs()
        with patched(*(stb.items() + [(D, "B1_OOD23", text)])):
            err = refused(D.reproductions)
        msg = str(err or "")
        check("the text's constants (the plain reading's) under P7's name: refused before any GPU pass (no loader "
              "pre-flight, no pass)", err is not None and stb.preflights == 0 and not [x for x in stb.passes
                                                                                      if x[1] == "ood23"],
              (err, stb.preflights, stb.passes))
        check("  the refusal names both readings of the seed and says the constant is the plain reading's",
              "seed 0: P7's statistic 0.149603, plain 0.256235" in msg and
              "the registered p7 constant 0.2562 is not V3F's p7 reading (it is the plain reading's value)" in msg and
              "target-species agnostic AP, OtherPlant boxes as ignore regions" in msg and
              "over every ground-truth box" in msg, msg)
        check("  and nothing was recorded", not (D.equiv_dir() / "b1_ood23.json").exists())
    with inc("inc_mis2"):
        b1_world()
        stb = Stubs()
        wrong = {"p7": D.B1_OOD23["p7"], "plain": (0.2562, 0.2518, 0.2598)}
        with patched(*(stb.items() + [(D, "B1_OOD23", wrong)])):
            err = refused(D.reproductions)
        msg = str(err or "")
        check("a wrong plain constant refuses too (seed 1's 0.2518), naming seed 1's two readings, no other seed's",
              err is not None and "seed 1: P7's statistic 0.159426, plain 0.251712" in msg and
              "the registered plain constant 0.2518 is not V3F's plain reading" in msg and "reading's value" not in msg
              and "seed 0:" not in msg and not stb.passes, msg)
    with inc("inc_mis3"):
        b1_world()
        stb = Stubs()
        swapped = {"p7": D.B1_OOD23["plain"], "plain": D.B1_OOD23["p7"]}
        with patched(*(stb.items() + [(D, "B1_OOD23", swapped)])):
            err = refused(D.reproductions)
        msg = str(err or "")
        check("the two readings' constants swapped: refused, each constant named as the other reading's value",
              err is not None and "(it is the plain reading's value)" in msg and "(it is the p7 reading's value)" in msg
              and msg.count("seed ") == 3, msg)


def t_first():
    print("item 2: in the reopened pre-flight's world (e1a_dev and e3a_dev superseded, so due again), the constants "
          "are checked first: a refusal moves no record and runs no pass of any reproduction")
    old = dict(D.code_shas(), ds1="ffafd22fea598b693bb88dca78c667a313f4b6990520f1eb06566a56040ae7b5")
    text = {"p7": tuple(round(x, 4) for x in PLAIN_LIVE), "plain": tuple(round(x, 4) for x in PLAIN_LIVE)}
    with inc("inc_first") as d:
        for s in D.SEEDS:
            v3f_b1(s, PLAIN_LIVE[s])
            e3 = TS.arm_dir("A", s) / TS.file_names(TS.EXAM)[0]
            e3.parent.mkdir(parents=True, exist_ok=True)
            e3.write_text(json.dumps({"stage1": {"agnostic_map50_95": 0.8568578547397661}, "species_map50_95": 0.8859}))
        recs = [equiv("e1a_dev", code=old), equiv("e3a_dev", code=old)]
        equiv("md_identity")
        sup = tuple({"deviation": 1, "path": str(p.relative_to(d)), "sha256": D._sha(p), "why": "test"} for p in recs)
        before = [p.read_bytes() for p in recs]
        stb = Stubs()
        with patched(*(stb.items() + [(D, "B1_OOD23", text), (D, "SUPERSEDED", sup)])):
            err = refused(D.reproductions)
        check("the text's constants under P7's name: refused with no pass at all (no E1-A dev pass, no loader "
              "pre-flight, no B'1 pass)", err is not None and stb.passes == [] and stb.preflights == 0,
              (err, stb.passes, stb.preflights))
        check("  the superseded records were not moved (still at their paths, byte for byte; no superseded/ folder)",
              [p.read_bytes() if p.is_file() else None for p in recs] == before and
              not (D.equiv_dir() / "superseded").exists(), sorted(os.listdir(str(D.equiv_dir()))))
        stb = Stubs()
        with patched(*(stb.items() + [(D, "SUPERSEDED", sup)])):
            err = refused(D.reproductions)
        check("  with the registered constants the same world moves both records, then takes E1-A's three dev passes "
              "before B'1's three ood23 passes", err is None and [x[1] for x in stb.passes] == ["dev"] * 3 +
              ["ood23"] * 3 and (D.equiv_dir() / "superseded" / "dev1" / "e1a_dev.json").is_file(), (err, stb.passes))


def t_band():
    print("item 2: each pass's two readings within DS.7.1's rule (1e-6 passed, 0.002 a recorded deviation)")
    cases = [("p7_0.001", {"dp7": 0.001}, "deviation"), ("p7_0.003", {"dp7": 0.003}, "refused"),
             ("plain_0.003", {"dplain": 0.003}, "refused"), ("plain_5e-4", {"dplain": 5e-4}, "deviation"),
             ("o_differs", {"o_val": 0.5}, "passed")]
    for tag, kw, want in cases:
        with inc("inc_band_" + tag):
            b1_world()
            stb = Stubs(**kw)
            with patched(*stb.items()):
                err = refused(D.reproductions)
            p = D.equiv_dir() / "b1_ood23.json"
            got = "refused" if err is not None else json.loads(p.read_text()).get("status")
            check("%s: %s" % (tag, want), got == want and (want != "refused" or not p.exists()), (got, err))
    with inc("inc_band_diff"):
        b1_world()
        stb = Stubs(dplain=5e-4)
        with patched(*stb.items()):
            D.reproductions()
        per = json.loads((D.equiv_dir() / "b1_ood23.json").read_text())["per"]["0"]
        check("  the seed's diff is the larger of the two readings' differences, each kept",
              abs(per["diff"] - 5e-4) < 1e-12 and per["readings"]["p7"]["diff"] == 0.0 and
              abs(per["readings"]["plain"]["diff"] - 5e-4) < 1e-12, per)


def t_readings():
    print("item 2: V3F's two readings from its score file (P7: the target arrays, checked by sha256; plain: stage1)")
    with inc("inc_read"):
        t, o = arrays(1), arrays(2)
        d = v3f_b1(0, 0.25, t=t, o=o)
        got = D.v3f_b1_readings(0)
        want_t = D.F.statistic_value(t, "agnostic")
        check("p7 is the target arrays' statistic (never the OtherPlant ones), plain the file's stage-1 agnostic AP",
              abs(got["p7"] - want_t) < 1e-15 and abs(want_t - D.F.statistic_value(o, "agnostic")) > 1e-6 and
              got["plain"] == 0.25 and got["sha256"] == D._sha(d / "ood23.json"), (got, want_t))
        doc = json.loads((d / "ood23.json").read_text())
        (d / "ood23.json").write_text(json.dumps(dict(doc, stage1={})))
        e_plain = refused(D.v3f_b1_readings, 0)
        (d / "ood23.json").write_text(json.dumps(dict(doc, p7=dict(doc["p7"], sha256="0" * 64))))
        e_sha = refused(D.v3f_b1_readings, 0)
        check("refused: no stage-1 agnostic AP in the file; p7 arrays that do not hash as recorded",
              e_plain is not None and e_sha is not None, (e_plain, e_sha))


# ------------------------------------------------------------------ item 3: the records taken by the earlier code
def t_supersede():
    print("item 3: e1a_dev and e3a_dev taken under the earlier code are moved aside and taken again")
    old = dict(D.code_shas(), ds1="ffafd22fea598b693bb88dca78c667a313f4b6990520f1eb06566a56040ae7b5")
    with inc("inc_sup") as d:
        p1 = equiv("e1a_dev", code=old)
        p3 = equiv("e3a_dev", code=old)
        b1 = equiv("b1_ood23")
        equiv("md_identity")
        e_live = refused(D.require_reproductions, ("e1a_dev", "e3a_dev"))
        check("the live state: records taken by the earlier code refuse every later score (require_reproductions)",
              e_live is not None and "the code changed since" in str(e_live), e_live)
        sup = tuple({"deviation": 1, "path": str(p.relative_to(d)), "sha256": D._sha(p), "why": "test"}
                    for p in (p1, p3))
        b1_bytes, old_bytes = b1.read_bytes(), {p.name: p.read_bytes() for p in (p1, p3)}
        for s in D.SEEDS:
            e3 = TS.arm_dir("A", s) / TS.file_names(TS.EXAM)[0]
            e3.parent.mkdir(parents=True, exist_ok=True)
            e3.write_text(json.dumps({"stage1": {"agnostic_map50_95": 0.8568578547397661}, "species_map50_95": 0.8859}))
        stb = Stubs()
        with patched(*(stb.items() + [(D, "SUPERSEDED", sup)])):
            err = refused(D.reproductions)
        moved = D.equiv_dir() / "superseded" / "dev1"
        check("the pre-flight moves both aside and takes them again: then require_reproductions passes",
              err is None and refused(D.require_reproductions, ("e1a_dev", "e3a_dev")) is None, err)
        check("  the moved records keep their bytes (hard-linked, never deleted) under equivalence/superseded/dev1/",
              all((moved / n).read_bytes() == b for n, b in old_bytes.items()), sorted(os.listdir(str(moved)))
              if moved.is_dir() else None)
        man = json.loads((moved / "MANIFEST.json").read_text()) if (moved / "MANIFEST.json").is_file() else {}
        check("  MANIFEST.json records each move: path, sha256, size, when, why, the deviation's file",
              [m["path"] for m in man.get("moves") or []] == [x["path"] for x in sup] and
              all(m["sha256"] == x["sha256"] and m["size"] == len(old_bytes[pathlib.Path(x["path"]).name]) and
                  m["deviation"]["file"] == "docs/prereg/DS1_dev1_2026-10-30.md" and m.get("moved_utc")
                  for m, x in zip(man["moves"], sup)) if man else False, man)
        check("  a record the deviation does not name stays as it was (b1_ood23)", b1.read_bytes() == b1_bytes)
        with patched((D, "SUPERSEDED", sup)):
            again = D.supersede()
        check("  run again, nothing moves (both moved_before), the retaken records untouched",
              [x["status"] for x in again] == ["moved_before", "moved_before"] and p1.is_file() and
              D._sha(p1) != sup[0]["sha256"], again)
    with inc("inc_sup2") as d:
        p1 = equiv("e1a_dev", code=old)
        sup = ({"deviation": 1, "path": str(p1.relative_to(d)), "sha256": "f" * 64, "why": "test"},)
        with patched((D, "SUPERSEDED", sup)):
            got = D.supersede()
        check("a file at the path with another sha256 is not the record named: it stays, reported other",
              [x["status"] for x in got] == ["other"] and p1.is_file(), got)
        sup = ({"deviation": 1, "path": str(p1.relative_to(d)), "sha256": D._sha(p1), "why": "test"},)
        dst = D.equiv_dir() / "superseded" / "dev1" / p1.name
        dst.parent.mkdir(parents=True)
        os.link(str(p1), str(dst))
        with patched((D, "SUPERSEDED", sup)):
            got = D.supersede()
        check("a move cut off after its link is completed (the original unlinked, the copy kept, recorded)",
              [x["status"] for x in got] == ["moved"] and not p1.exists() and dst.is_file(), got)
        p1.write_bytes(b"{}")
        dst.write_bytes(b"other bytes")
        with patched((D, "SUPERSEDED", sup)):
            e_dst = refused(D.supersede)
        check("  a folder copy with other bytes than the record refuses (never overwritten)", e_dst is not None, e_dst)
    check("the live entries name the two records of job 50026958 by their sha256, under deviation 1",
          [(x["path"], x["sha256"][:12], x["deviation"]) for x in D.SUPERSEDED] ==
          [("ds1/v1/equivalence/e1a_dev.json", "a3d247985908", 1), ("ds1/v1/equivalence/e3a_dev.json", "2ea945605f06", 1)])


# ------------------------------------------------------------------ item 4: the reopen (the live state)
def live_item():
    """score:POWER:draft_r001 as the lab's state held it on 2026-10-30 after the second refusal."""
    first = {"proposal": "b568033a258c115ca1f44d685e1cc8fd", "gpuh": 0.1144, "how": "settled"}
    second = {"proposal": "c9b407ddb437f7847e002974e2040010", "gpuh": 0.0333, "how": "settled"}
    third = {"proposal": "9fe0bb091892ae08c4ae819ef7cd2ebc", "gpuh": 0.2125, "how": "settled"}
    hist = {"attempts": [first, second], "build_job_gpuh": None, "estimate": 1.5, "exp": None, "fails": 2,
            "lever": "L33", "marker": "DS1 Revision 2 (2026-10-29)", "phase": 1, "prior_gpuh": 0.1144,
            "proposal": second["proposal"], "reopened_utc": "2026-10-30T02:32:00Z", "settled": 0.0333,
            "spend": "settled", "started_utc": "2026-10-29T18:17:04Z", "state": "failed"}
    return {"attempts": [first, second, third], "build_job_gpuh": None, "estimate": 1.5, "exp": None, "fails": 2,
            "lever": "L33", "phase": 1, "prior_gpuh": 0.3602, "proposal": "41d69870631f113a238038b57ed7ea1b",
            "reopen_history": [hist], "reopened": "DS1 Revision 2 (2026-10-29)",
            "reopened_utc": "2026-10-30T02:32:00Z", "settled": 0.0969, "spend": "settled",
            "started_utc": "2026-10-30T03:12:00Z", "state": "failed"}


def reopen_run(st, deployed):
    run = types.SimpleNamespace(st=st, dom=LS.load_domain("weed"), ev=None, utc="2026-10-30T09:00:00Z")
    run._ds1 = types.MethodType(ST.StreamRun._ds1, run)
    run.ledger, run.cards = [], []
    run._ds1_tree_file = lambda rel, sha=None, marker=None: (rel, sha) in deployed
    run._failed_ids = types.MethodType(ST.StreamRun._failed_ids, run)
    run._v3fam_fold = ST.StreamRun._v3fam_fold
    run._ledger = lambda kind, **kw: run.ledger.append((kind, kw))
    run._card = lambda kind, title, body, **kw: run.cards.append((title, body))
    return run


def t_reopen():
    print("item 4: the live score:POWER:draft_r001 reopened under the deviation, its spend kept")
    dom = LS.load_domain("weed")["ds1"]
    blk = next((b for b in dom.get("reopens") or [] if b.get("marker") == "DS1 deviation 1 (2026-10-30)"), {})
    dv = D.DEVIATIONS[0]
    check("the domain's reopen block names the power pre-flight, the deviation's file and sha256, and a time after "
          "the live attempt (2026-10-30T03:12:00Z)", blk.get("items") == [POWER] and blk.get("file") == dv["file"]
          and blk.get("sha256") == dv["sha256"] and str(blk.get("before_utc")) > "2026-10-30T03:12:00Z", blk)
    rev2 = next(b for b in dom["reopens"] if b["marker"] == "DS1 Revision 2 (2026-10-29)")
    both = {(rev2["file"], rev2["sha256"]), (dv["file"], dv["sha256"])}
    st = {"failed_ids": ["b568033a258c115ca1f44d685e1cc8fd", "c9b407ddb437f7847e002974e2040010",
                         "9fe0bb091892ae08c4ae819ef7cd2ebc"], "ds1": {"items": {POWER: live_item()}, "cards": {}}}
    run = reopen_run(st, {(rev2["file"], rev2["sha256"])})
    check("not reopened while the deployed tree holds Revision 2 but not the deviation (the live state)",
          ST.StreamRun._ds1_reopen(run) == [] and st["ds1"]["items"][POWER]["state"] == "failed")
    run = reopen_run(st, both)
    got = ST.StreamRun._ds1_reopen(run)
    rec = st["ds1"]["items"][POWER]
    check("reopened once deployed: state missing, fails 0, no proposal, marked with the deviation's marker",
          got == [POWER] and rec["state"] == "missing" and rec["fails"] == 0 and rec["proposal"] is None and
          rec["reopened"] == "DS1 deviation 1 (2026-10-30)", (got, rec))
    check("  its spend kept: prior_gpuh 0.3602 + 0.0969 = 0.4571, the attempt recorded; the proposal id failed (a "
          "new one is taken)", abs(rec["prior_gpuh"] - 0.4571) < 1e-9 and rec["attempts"][-1] == {
              "proposal": "41d69870631f113a238038b57ed7ea1b", "gpuh": 0.0969, "how": "settled"} and
          "41d69870631f113a238038b57ed7ea1b" in st["failed_ids"], (rec.get("prior_gpuh"), st["failed_ids"][-3:]))
    check("  the history keeps both reopenings in order (Revision 2, then the deviation); one card, one ledger line",
          [h["marker"] for h in rec["reopen_history"]] == ["DS1 Revision 2 (2026-10-29)",
                                                           "DS1 deviation 1 (2026-10-30)"] and
          len(run.cards) == 1 and "reopened" in run.cards[0][0] and run.ledger[0][0] == "ds1_reopened",
          ([h.get("marker") for h in rec["reopen_history"]], run.cards))
    rec.update(state="failed", started_utc="2026-10-30T03:40:00Z", proposal="pidX")
    check("  never twice under one marker", ST.StreamRun._ds1_reopen(run) == [] and rec["state"] == "failed")
    st["ds1"]["items"][POWER] = {"state": "failed", "started_utc": "2026-10-31T00:00:00Z", "proposal": "pidY",
                                 "reopened": "DS1 Revision 2 (2026-10-29)"}
    check("  an attempt started after the deviation is not reopened by it", ST.StreamRun._ds1_reopen(run) == [])
    fs = {"status": {"phases": {"1": {"revisions": 1, "new_batches": 0, "sources": {}}},
                     "md_s0": {"drafts": 1, "draft": "draft_r001", "power": False, "frozen": False}, "arms": {},
                     "sheets": {"source_batches": 0, "ln_batches": 0, "mtw_batches": 0, "rendered": ["source_001"],
                                "audit_pending": [], "audit_sets": {}}, "scores": {}, "verdicts": {},
                     "long": {"arms": {}, "pair": None}},
          "items": {"data:1:screen": "missing", POWER: "missing"}, "today": "2026-10-30", "phase1_gb": 2.0,
          "audit_render_for": None}
    _o, _w, c = DS.ds1_plan(LS.load_domain("weed")["ds1"], fs)
    check("  reopened, the plan proposes the power pre-flight (L33 POWER) again",
          [x["item"] for x in c["maint"]][:1] == [POWER], [x["item"] for x in c["maint"]])


# ------------------------------------------------------------------ the gate
def t_gate():
    print("the gate: no DS1 lever until the deployed tree holds the deviation's file by its pinned sha256")
    cfg = LS.load_domain("weed")["ds1"]
    files = [(cfg["amendment_file"], cfg["amendment_sha256"])] + [(r["file"], r["sha256"]) for r in
                                                                  cfg["revisions"] + cfg.get("deviations", [])]
    check("the domain pins the deviation (ds1.deviations) as inc2.ds1.DEVIATIONS does",
          [(r["file"], r["sha256"]) for r in cfg.get("deviations") or []] ==
          [(r["file"], r["sha256"]) for r in D.DEVIATIONS])

    def stage(present):
        run = types.SimpleNamespace(st={"ds1": {"items": {}, "cards": {}}}, dom=LS.load_domain("weed"), ev=None,
                                    utc="2026-10-30T09:00:00Z")
        run._ds1 = types.MethodType(ST.StreamRun._ds1, run)
        run._ds1_reopen = lambda: []
        run._artifact = lambda *a, **k: {"timestamp": True}
        run._ds1_spend = lambda *a, **k: 0.0
        run._ds1_tree_file = lambda rel, sha=None, marker=None: (rel, sha) in present if sha is not None else True
        return ST.StreamRun._ds1_stage(run, {}, None)

    check("every pinned file deployed: the amendment input is true", stage(set(files))["amendment"] is True)
    check("  the deviation's file missing (or another sha256): false, and DR0's gate stays shut",
          stage(set(files[:-1]))["amendment"] is False and
          DS._ds1_gate(None, {"ds1": {"amendment": False}}, cfg)[0] is False)


# ------------------------------------------------------------------ the registration
def t_registration():
    print("the registration: the deviation's byte copy, its pins, its proofs and stamp file")
    dv = D.DEVIATIONS[0] if D.DEVIATIONS else {}
    f = GIT / dv.get("file", "missing")
    data = f.read_bytes() if f.is_file() else b""
    check("DS1 has deviation 1; its byte copy hashes to its pin", [d["n"] for d in D.DEVIATIONS] == [1] and data and
          hashlib.sha256(data).hexdigest() == dv.get("sha256"), dv)
    cl = (GIT / "docs" / "CONTINUOUS_LOOP.md").read_bytes()
    check("  it is docs/CONTINUOUS_LOOP.md's deviation section byte for byte, to the end of the file", data and
          cl.endswith(data) and data.startswith(b"## Deviation (2026-10-30): DS1 deviation 1"))
    rev2 = (GIT / D.REVISIONS[1]["file"]).read_bytes()
    check("  appended after Revision 2's section, which stays byte for byte (its heading to the blank line before "
          "the next heading)", cl.endswith(rev2 + b"\n" + data))
    stamp = GIT / dv.get("stamp_file", "missing")
    text = stamp.read_text() if stamp.is_file() else ""
    check("its proofs stamp the byte copy and the stamp file (OpenTimestamps headers)",
          D.ots_digest(str(f) + ".ots") == dv.get("sha256") and
          D.ots_digest(str(stamp) + ".ots") == (hashlib.sha256(stamp.read_bytes()).hexdigest() if text else None))
    check("  the stamp file names its sha256 and commit, the amendment's and both Revisions' sha256s (unchanged), "
          "and the superseded records' sha256s", dv.get("sha256") and dv["sha256"] in text and dv.get("commit") and
          dv["commit"] in text and D.PREREG_SHA256 in text and all(r["sha256"] in text for r in D.REVISIONS) and
          all(x["sha256"] in text for x in D.SUPERSEDED))
    blk = next((b for b in LS.load_domain("weed")["ds1"].get("reopens") or [] if b.get("file") == dv.get("file")), {})
    check("  the reopen block's before_utc is the stamp file's written_utc", blk.get("before_utc") and
          ("written_utc: %s" % blk["before_utc"]) in text, blk.get("before_utc"))


def main():
    for fn in (t_live, t_mislabel, t_first, t_band, t_readings, t_supersede, t_reopen, t_gate, t_registration):
        try:
            fn()
        except Exception as e:                   # noqa: BLE001
            check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-2500:]))
    print("\n%d failure(s), 0 skipped: %s" % (len(FAILURES), "; ".join(FAILURES[:5])))
    shutil.rmtree(str(TMP), ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
