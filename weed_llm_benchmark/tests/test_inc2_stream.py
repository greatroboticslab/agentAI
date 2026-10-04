#!/usr/bin/env python3
"""The stream builder of the continuous loop (inc2/stream.py;
docs/CONTINUOUS_LOOP.md §3.4-3.8, §9 group E acceptance).

No cluster, no training, no network: inc.driver's FakeBackend 'runs' every
spec through a synthetic executor that writes run.json, scores/<exam>.json,
weights/final.pt and the Protocol v3 scorer sidecar the way inc2.train does.
Images are small synthetic PNGs whose dHash is controlled (a 9 x 8 pattern
of distinct levels, upscaled), so every copy kind is planted exactly.

Pinned:
  * the disposition rule (§3.5 [review]) on the recorded realloop_v1 and
    pilot_v3 ledgers (tests/fixtures/inc_replay): data / species / flips as
    the contract's worked example, s03 read by the rule (its truth arm says
    helps); pilot_v3's Bswap is 'data' although its blame is 'recipe' (S19);
    recipe-only, HOLD and truth-helps cases; the permutation test; a
    v3-unavailable step keeps the pinned guards;
  * the ledger: hash chain verified, a broken chain refused, a partial last
    line cut back;
  * init: P_0 is base_v2's bytes, M = ceil(0.10 |base|) or given, the
    prospective record precedes any build, a stream is defined once;
  * the cut: exactly M; deterministic under stable_int; whole near-duplicate
    units; target images only; holds honoured (a person's release lifts one;
    the rows a licence release lifts are research_only in the rows sidecar
    unless it records the person's --licence and --not-research-only);
    refused, unevidenced, OtherPlant-only and base-copy rows never cut;
    planted exact, 3-bit, 6-bit, hflip and rot90 copies of a dev image and a
    masked row whose original is a dev copy refused by GuardV2 at the cut;
    one source first; the 60 % species cap with two sources; a split capture
    group recorded and its remainder cut first; capture groups that cannot
    sum to M split by unit (S24); units that cannot sum to M refused with a
    reason (cut_refusal.json), never a silent wait;
  * build: exp.json passes the pinned validate_definition and
    check_definition_data; replay full, gate net, clean false, finals dev
    and imageweeds, truth on, the arm stamp; one sbatch array;
  * end to end on FakeBackend: segment 1 (ACCEPT, REJECT data) -> P_1 = P_0
    + the accepted increment, disjoint; the rejected one quarantined and a
    re-harvested copy refused by dHash; an unfinished segment refused;
    segment 2 (HOLD, recipe, data, data) -> returns counted; segment 3 cuts
    the returned images first and a second non-accept makes them neutral;
    Protocol v3 accepts a species-only pinned REJECT (discordant, recorded);
    milestone build and the 5 v 5 comparison (helps, then hurts -> rollback
    to the last good pool, suspect increments, bisect: hurts quarantined,
    helps returned); the boundary check; Stage C feasibility; withdraw; fork;
  * governance of the verbs: rollback outside the recommendation needs a
    person, a source quarantine needs a cited diagnosis, lifting it needs a
    person, commit without inc2.gate3 refuses;
  * the single-writer lease; the CLI reads back the L18 argv;
  * queue_summary.json: its schema, and test blindness: every test and
    ImageWeeds score perturbed -> the same summary;
  * run_inc2_build.sh: verbs, INC_JOB_SCRIPT, the module drift check, the
    provenance record and the advance of the experiment a stream verb built;
    inc2.baseline rescore-native runs under its own provenance name and is
    never advanced.

Run:  python3 tests/test_inc2_stream.py
"""
import collections
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc2_stream_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
os.environ["INC_SCORER_TESTING"] = "1"
os.environ.pop("INC_JOB_SCRIPT", None)
PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import driver as D  # noqa: E402
from weed_optimizer_framework.tools.inc import gate as G  # noqa: E402
from weed_optimizer_framework.tools.inc2 import stream as ST  # noqa: E402

try:
    from weed_optimizer_framework.tools.inc2 import guard as GA  # noqa: E402
except ImportError as _e:                                        # group A not installed
    GA = None
    print("NOTE: inc2.guard is not installed (%s); a GuardV2 double from funnel.leak is used" % _e)

FAILURES = []
SPECIES = list(C.CLASS_NAMES[:C.OTHER_PLANT])
FIXTURES = PKG_ROOT / "tests" / "fixtures" / "inc_replay"


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def raises(fn, exc=Exception, contains=None):
    try:
        fn()
    except exc as e:
        if contains and contains not in str(e):
            print("       raised %s without %r: %s" % (type(e).__name__, contains, e))
            return False
        return True
    return False


# ------------------------------------------------------------------ images
def pattern(seed):
    rng = np.random.default_rng(seed)
    return np.stack([rng.permutation(9) * 25 + 10 for _ in range(8)]).astype(np.uint8)


def save_pattern(a, path, noise=0, seed=0):
    big = np.kron(a, np.ones((8, 8), dtype=np.uint8)).astype(np.int16)
    if noise:
        big = big + np.random.default_rng(seed).integers(-noise, noise + 1, big.shape)
    big = np.clip(big, 0, 255).astype(np.uint8)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.stack([big] * 3, -1)).save(path)
    return path


def near_pattern(a, n_swaps, seed=0):
    b = a.copy()
    rng = np.random.default_rng(seed)
    for _ in range(n_swaps):
        r, c = int(rng.integers(0, 8)), int(rng.integers(0, 8))
        b[r, c], b[r, c + 1] = b[r, c + 1], b[r, c]
    return b


def hashes(path):
    if GA is not None:
        return GA.image_hashes(path)
    from weed_optimizer_framework.tools.funnel import leak as L
    with Image.open(path) as im:
        im.load()
        v = {k: int(x) for k, x in L.dhash_variants(im).items()}
    return v["id"], v


class GuardDouble:
    """GuardV2's never-train and base-copy checks (used only without inc2.guard)."""

    def __init__(self, eval_entries, base_entries):
        from weed_optimizer_framework.tools.near_dup import NearHashIndex
        self.ev, self.base = NearHashIndex(), NearHashIndex()
        for h, s, k in eval_entries:
            self.ev.add(int(h), (s, k), max_bits=6)
        for h, s, k in base_entries:
            self.base.add(int(h), (s, k), max_bits=6)

    def check(self, dhash, variants=None):
        if dhash is None or not variants:
            return "unhashable", None
        if self.ev.find(int(dhash)) is not None:
            return "near_eval_v2", None
        if any(self.ev.find(int(v)) is not None for k, v in variants.items() if k != "id"):
            return "near_eval_variant", None
        if any(self.base.find(int(v)) is not None for v in variants.values()):
            return "base_copy", None
        return None, None


def make_guard(eval_paths, base_rows):
    ev = [(hashes(p)[0], "dev", "dev_%d" % i) for i, p in enumerate(eval_paths)]
    base = [(hashes(r["image"])[0], "base_v2", r["key"]) for r in base_rows]
    return GA.GuardV2(ev, base) if GA is not None else GuardDouble(ev, base)


def write_label(path, boxes):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for i, c in enumerate(boxes):
            fh.write("%d %.6f %.6f 0.100000 0.100000\n" % (c, 0.15 + 0.1 * (i % 7), 0.2 + 0.08 * (i % 5)))
    return path


# ------------------------------------------------------------------- world
WORLD = TMP / "world"
SEED = [1000]


def next_seed():
    SEED[0] += 1
    return SEED[0]


def make_base(n_core=30, tsw_sessions=(5, 4, 6, 3, 5, 4)):
    """base_v2: n_core 'train_core' rows and tsw22 rows in whole sessions."""
    rows = []
    for i in range(n_core):
        p = save_pattern(pattern(next_seed()), WORLD / "base" / ("tc_%03d.png" % i))
        lab = write_label(WORLD / "base" / ("tc_%03d.txt" % i), [i % 12, (i + 5) % 12])
        rows.append({"image": str(p), "label": str(lab), "sha256": C.sha256_file(p),
                     "label_sha256": C.sha256_file(lab), "source": "cwd12/train", "session": "s%d" % (i // 6),
                     "key": "tc__%03d" % i})
    for si, n in enumerate(tsw_sessions):
        for j in range(n):
            p = save_pattern(pattern(next_seed()), WORLD / "base" / ("tsw22_%d_%d.png" % (si, j)))
            lab = write_label(WORLD / "base" / ("tsw22_%d_%d.txt" % (si, j)), [2, 8, 12])
            rows.append({"image": str(p), "label": str(lab), "sha256": C.sha256_file(p),
                         "label_sha256": C.sha256_file(lab), "source": "3seasonweeddet10/data2022",
                         "session": "sess%02d" % si, "key": "tsw22__sess%02d_%03d" % (si, j)})
    path = WORLD / "base_v2.jsonl"
    C.write_manifest(path, rows)
    return path, C.read_manifest(path)


class Step1World:
    """An INC_DIR/<name> directory laid out as step1_stream's (queue, events,
    evidence index, batches)."""

    def __init__(self, name):
        self.root = pathlib.Path(C.INC_DIR) / name
        (self.root / "queue").mkdir(parents=True, exist_ok=True)
        (self.root / "index").mkdir(parents=True, exist_ok=True)
        self.evidence = {}
        self.rows = []

    def add(self, source, batch, species, n=1, capture_group="", pat=None, noise=0, admission="whole",
            unmasked_pattern=None, kind=None, holds=(), evidenced=True, research_only=False, other=0,
            l1=0, l2=0, key=None, image_from=None):
        out = []
        for i in range(n):
            k = key or "%s__%04d" % (source, len(self.rows))
            d = self.root / "files" / source
            a = pat if pat is not None else pattern(next_seed())
            if image_from is not None:
                img = d / ("%s.png" % k)
                img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(image_from, img)
            else:
                img = save_pattern(a, d / ("%s.png" % k), noise=noise, seed=next_seed())
            unm = img
            if admission == "masked":
                unm = save_pattern(unmasked_pattern if unmasked_pattern is not None else a,
                                   d / ("%s.orig.png" % k), noise=noise, seed=next_seed())
            boxes = [s for s, c in enumerate(species) for _ in range(c)] + [12] * other
            lab = write_label(d / ("%s.txt" % k), boxes)
            hs = hashes(unm)[0]
            hm = hashes(img)[0] if admission == "masked" else None
            row = {"format": "inc2-stream-queue/1", "key": k, "batch": batch, "source": source, "group": None,
                   "capture_group": capture_group, "l1": l1, "l2": l2,
                   "kind": kind or ("cwd12" if sum(species) else "other"), "score": 0.5,
                   "species_boxes": list(species), "other_boxes": other, "admission": admission,
                   "n_masked": 1 if admission == "masked" else 0, "evidenced": evidenced, "lab_group": None,
                   "licence": "CC-BY-4.0", "research_only": research_only, "prior": None,
                   "hold_until": holds[0] if holds else None, "holds": list(holds),
                   "hold_deadline": {h: "2026-10-01" for h in holds}, "verifier": "v1", "reference": "v1",
                   "stream_pins_sha": "p" * 64, "image": str(img), "label": str(lab),
                   "sha256": C.sha256_file(img), "label_sha256": C.sha256_file(lab), "session": capture_group,
                   "unmasked_image": str(unm), "unmasked_sha256": C.sha256_file(unm), "dhash": int(hs),
                   "dhash_masked": None if hm is None else int(hm), "masked_area_frac": 0.1 if hm else 0.0,
                   "input": "intake", "supersedes": None}
            self.rows.append(row)
            if evidenced:
                self.evidence[source] = self.evidence.get(source, 0) + sum(species)
            out.append(row)
            key = None
        return out

    def commit(self, batches=None):
        """Write queue.jsonl (every row once), the evidence index and batch.json files."""
        with open(self.root / "queue" / "queue.jsonl", "w") as fh:
            for r in self.rows:
                fh.write(json.dumps(r, sort_keys=True) + "\n")
        (self.root / "index" / "evidence.json").write_text(json.dumps({"applied": [], "data": self.evidence}))
        for b in sorted({r["batch"] for r in self.rows}):
            bd = self.root / "batches" / b
            bd.mkdir(parents=True, exist_ok=True)
            if not (bd / "batch.json").exists():
                (bd / "batch.json").write_text(json.dumps({"committed_utc": "2026-09-2%sT00:00:00Z" % b[-1]}))

    def event(self, **e):
        with open(self.root / "queue" / "events.jsonl", "a") as fh:
            fh.write(json.dumps(e, sort_keys=True) + "\n")


# --------------------------------------------------------------- executor
NOISE = {0: 0.0, 1: 0.001, 2: -0.001, 3: 0.0005, 4: -0.0005}
# kind -> (cand effect, null effect) of an incremental step, and its cold effect
STEP_EFFECT = {"good": (0.02, 0.0), "bad": (-0.06, 0.0), "flat": (0.0, 0.0), "recipe": (-0.02, -0.02),
               "sneaky": (0.02, 0.0), "rare": (0.02, 0.0)}
COLD_EFFECT = {"good": 0.01, "bad": -0.03, "flat": 0.0, "recipe": 0.0, "sneaky": -0.05, "rare": 0.0}
RARE_SPECIES, RARE_DROP, RARE_SE = "PricklySida", 0.065, 0.03
FACTOR = {"dev": 1.0, "imageweeds": 0.8, "test": 1.05}
SE_DEFAULT = 0.004
PERTURB = {}          # exam -> added to every score of that exam (test blindness)


def kind_of_key(key):
    head = key.split("__")[0]
    k = head.split("_")[0]
    return k if k in COLD_EFFECT else None


class Executor:
    def __init__(self, m):
        self.m = m
        self.runs = []
        self.no_sidecar = set()

    def __call__(self, spec):
        out = pathlib.Path(spec["out_dir"])
        rid, kind = spec["run_id"], spec["kind"]
        self.runs.append(rid)
        if (out / "run.json").exists():
            (out / "run.json").unlink()
        w = out / "weights" / "final.pt"
        w.parent.mkdir(parents=True, exist_ok=True)
        if kind == "final":
            if w.is_symlink() or w.exists():
                w.unlink()
            os.symlink(spec["init"], w)
        else:
            w.write_bytes(("weights of %s %s" % (spec["exp"], rid)).encode())
        for exam in spec["exams"]:
            v, pc = self.values(spec, exam)
            self._score(out / "scores" / ("%s.json" % exam), exam, v, pc, w)
            if exam == "dev" and kind in ("base", "union", "cand", "soup") and rid not in self.no_sidecar:
                self._sidecar(out / "scores" / "dev.sidecar.json", v, w)
        (out / "run.json").write_text(json.dumps({"status": "done", "attempt": 1, "seconds": 100.0, "error": None,
                                                  "weights_sha256": C.sha256_file(w)}))
        return "COMPLETED"

    @staticmethod
    def parent(weights):
        s = json.loads((pathlib.Path(weights).parent.parent / "scores" / "dev.json").read_text())
        return s["map50_95"], s["per_class"]

    def step_kind(self, spec):
        tag = spec["run_id"].split("__")[1]
        name = tag.split("_", 1)[1]
        if name.startswith("D_"):
            return "good"
        sid = ST.sid_of_exp(spec["exp"])
        rows = C.read_manifest(ST.StreamPaths(sid).inc_manifest(name))
        kinds = collections.Counter(kind_of_key(r["key"]) for r in rows)
        return kinds.most_common(1)[0][0]

    def values(self, spec, exam):
        kind = spec["kind"]
        if kind in ("base", "union"):
            rows = C.read_manifest(spec["train_manifest"])
            eff = sum(COLD_EFFECT.get(kind_of_key(r["key"]), 0.0) for r in rows) / float(self.m)
            v = 0.40 + eff + NOISE[spec["recipe"]["seed"]]
            pc = {s: v for s in SPECIES}
        elif kind in ("cand", "null"):
            pv, ppc = self.parent(spec["init"])
            k = self.step_kind(spec)
            ce, ne = STEP_EFFECT[k]
            e = ce if kind == "cand" else ne
            n = NOISE[spec["recipe"]["seed"]]
            v = pv + e + n
            pc = {s: ppc[s] + e + n for s in SPECIES}
            if kind == "cand" and k == "rare":
                pc[RARE_SPECIES] -= RARE_DROP
        elif kind == "soup":
            vals = [self.parent(x) for x in spec["soup_of"]]
            v = float(np.mean([a for a, _ in vals])) + 0.001
            pc = {s: float(np.mean([b[s] for _, b in vals])) + 0.001 for s in SPECIES}
        else:
            pv, ppc = self.parent(spec["init"])
            v = pv * FACTOR[exam]
            pc = {s: ppc[s] * FACTOR[exam] for s in SPECIES}
        v += PERTURB.get(exam, 0.0)
        return v, pc

    @staticmethod
    def _score(path, exam, v, pc, weights):
        other = 0 if exam in ("dev", "test") else 25
        n_gt = {s: 40 for s in SPECIES}
        n_gt["OtherPlant"] = other
        per_class = dict(pc)
        if other:
            per_class["OtherPlant"] = 0.5 * v
        s = {"exam": exam, "scorer_sha256": "TEST-" + "5" * 64, "production": False,
             "deviations": ["LOCK.json not checked"],
             "manifest_sha256": hashlib.sha256(("m/" + exam).encode()).hexdigest(),
             "key_order_sha256": hashlib.sha256(("k/" + exam).encode()).hexdigest(),
             "weights_sha256": C.sha256_file(weights), "n_images": 20,
             "map50_95": v, "map50": min(1.0, v + 0.2), "agnostic_map50_95": v + 0.1,
             "agnostic_map50": min(1.0, v + 0.3), "per_class": per_class, "n_gt": n_gt,
             "image_correct": "1" * 20, "species_map50_95": v, "species_map50": v + 0.2}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(s))

    @staticmethod
    def _sidecar(path, v, weights):
        stamps = {"exam": "dev", "scorer_sha256": "TEST-" + "5" * 64,
                  "manifest_sha256": hashlib.sha256(b"m/dev").hexdigest(),
                  "key_order_sha256": hashlib.sha256(b"k/dev").hexdigest(), "n_images": 20}
        per = {s: {"se": RARE_SE if s == RARE_SPECIES else SE_DEFAULT, "n_valid": 1000, "n_gt": 40, "ap": v,
                   "mean": v, "p2_5": v - 0.01, "p97_5": v + 0.01} for s in SPECIES}
        doc = {"format": "inc2-scorer-sidecar/1", "exam": "dev", "weights_sha256": C.sha256_file(weights),
               "score": {"stamps": stamps, "production": False},
               "species_se": {"seed_text": "inc2/v3/species_se", "resamples": 1000, "per_species": per}}
        path.write_text(json.dumps(doc))


def drive(exp, fb, executor, max_iter=200):
    for _ in range(max_iter):
        ran = fb.run_pending(executor)
        before = len(fb.submissions)
        D.Driver(exp, backend=fb, quiet=True).advance()
        st = json.loads(D.Paths(exp).state.read_text())
        if st["done"] or (ran == 0 and len(fb.submissions) == before):
            return st
    return json.loads(D.Paths(exp).state.read_text())


class Clock:
    def __init__(self, t=1.8e9):
        self.t = t

    def __call__(self):
        return self.t


class BaselineDouble:
    """inc2.baseline build (milestones) on the pinned driver with a FakeBackend:
    the argv the stream passes, a baseline definition with the arm stamp."""

    def __init__(self, fb):
        self.fb = fb
        self.argvs = []

    def __call__(self, argv):
        from weed_optimizer_framework.tools.inc2 import recipes as RC
        self.argvs.append(list(argv))
        a = dict(zip(argv[1::2], argv[2::2])) if argv and argv[0] == "build" else {}
        exp, man = a["--exp"], a["--manifest"]
        seeds = [int(x) for x in a["--seeds"].split(",")]
        rows = C.read_manifest(man)
        arm = RC.resolve_arm(a.get("--arm", "n640"), repo=C.REPO, require_weights=False)
        defn = {"exp": exp, "type": "baseline", "testing": "--testing" in argv, "seeds": seeds,
                "decision_exam": "dev", "final_exams": a.get("--final-exams", "dev,imageweeds,test").split(","),
                "base": {"name": "pool", "manifest": man, "manifest_sha256": C.sha256_file(man),
                         "n_images": len(rows), "recipe": RC.table(arm["id"])["cold"]},
                "role": a.get("--role")}
        defn.update(RC.stamp(arm))
        D.Driver(exp, backend=self.fb, quiet=True).init(defn)
        return 0


# ------------------------------------------------------------------ tests
def test_run_as_main():
    """`python -m ...inc2.stream VERB` (every job script) runs the module as __main__: PKG and the other groups'
    modules must resolve then too (2026-09-29: a PKG derived from __name__ made stream init fail with 'inc2.recipes
    ... is not installed', job 47276839)."""
    import importlib.util
    import types
    spec = importlib.util.find_spec("weed_optimizer_framework.tools.inc2.stream")
    src = open(spec.origin).read()
    main_guard = 'if __name__ == "__main__":'
    mod = types.ModuleType("__main__")
    mod.__dict__.update(__name__="__main__", __package__="weed_optimizer_framework.tools.inc2", __spec__=spec,
                        __file__=spec.origin)
    exec(compile(src.replace(main_guard, "if False:"), spec.origin, "exec"), mod.__dict__)
    check("run as __main__ (python -m): PKG is the package, and _inc2 resolves recipes and gate3",
          src.count(main_guard) == 1 and mod.PKG == "weed_optimizer_framework.tools"
          and mod._inc2("recipes") is not None and mod._inc2("gate3") is not None, mod.PKG)


def test_rules():
    print("the disposition rule and the permutation test (pure functions)")
    p_reject = G.GateConfig().p_reject

    def ledger(path):
        return [json.loads(ln) for ln in open(path) if ln.strip()]

    rl = ledger(FIXTURES / "funnel" / "realloop_v1" / "ledger.jsonl")
    truth = {e["k"]: e["detail"]["verdict"] for e in rl if e.get("type") == "truth"}
    disp = {e["k"]: ST.dispose(ST.normalise(None, e), truth.get(e["k"]), p_reject)
            for e in rl if e.get("type") == "gate" and e["chain"] == "full"}
    check("realloop_v1 by the guard-based rule: s01 data, s02 species, s04 species, s05 flips, s06 species",
          [disp[k] for k in (1, 2, 4, 5, 6)] == ["data", "species", "species", "flips", "species"], disp)
    check("realloop_v1 s03 (P_data 0.11, truth arm 'helps') is not quarantined as data: the rule's exception "
          "applies (species)", truth[3] == "helps" and disp[3] == "species", (truth[3], disp[3]))
    blames = {e["k"]: e["decision"]["attribution"]["blame"] for e in rl if e.get("type") == "gate"}
    check("every realloop_v1 REJECT carries blame 'recipe', which the rule never reads",
          set(blames.values()) == {"recipe"}, blames)
    pv = ledger(FIXTURES / "pilot_v3" / "ledger.jsonl")
    ptruth = {e["k"]: e["detail"]["verdict"] for e in pv if e.get("type") == "truth"}
    pdisp = {e["step"]: ST.dispose(ST.normalise(None, e), ptruth.get(e["k"]), p_reject)
             for e in pv if e.get("type") == "gate" and e["chain"] == "full"}
    bswap = next(e for e in pv if e.get("type") == "gate" and e["chain"] == "full" and e["step"] == "Bswap")
    check("S19: pilot_v3's Bswap (P_data 0.00, blame 'recipe') is data; Breal is species; I1..I5 accepted",
          pdisp["Bswap"] == "data" and bswap["decision"]["attribution"]["blame"] == "recipe"
          and pdisp["Breal"] == "species" and all(pdisp[i] == "accepted" for i in ("I1", "I2", "I3", "I4", "I5")),
          pdisp)

    def rec(verdict, p, reg=True, sp=True, fl=True):
        return {"verdict": verdict, "p_data": p, "guards": {"regression": reg, "species": sp, "flips": fl}}
    check("recipe: only the regression guard failed", ST.dispose(rec("REJECT", 0.5, reg=False), None, 0.25) == "recipe")
    check("HOLD returns (guards pass, P between)", ST.dispose(rec("HOLD", 0.5), None, 0.25) == "hold")
    check("REJECT on P_data alone with truth helps -> truth_helps; without truth -> data",
          ST.dispose(rec("REJECT", 0.2), "helps", 0.25) == "truth_helps"
          and ST.dispose(rec("REJECT", 0.2), None, 0.25) == "data")
    check("data comes first even with a failed species guard", ST.dispose(rec("REJECT", 0.1, sp=False), "neutral", 0.25)
          == "data")
    unav = {"v3_applied": False, "commit_verdict": "REJECT", "reason": "no sidecar"}
    e5 = next(e for e in rl if e.get("type") == "gate" and e["k"] == 5)
    n5 = ST.normalise(unav, e5)
    check("a v3-unavailable step keeps the pinned verdict and guards and says why",
          n5["verdict"] == "REJECT" and not n5["v3_applied"] and n5["guards"]["flips"] is False
          and n5["v3_unavailable_reason"] == "no sidecar", n5)

    check("permutation test: 5 v 5 separated -> 1/252; reversed -> 1.0",
          abs(ST.permutation_p([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]) - 1 / 252.0) < 1e-12
          and ST.permutation_p([6, 7, 8, 9, 10], [1, 2, 3, 4, 5]) == 1.0)


def test_ledger():
    print("the hash-chained ledger")
    p = TMP / "ledger_test" / "ledger.jsonl"
    L = ST.Ledger(p)
    L.read()
    for i in range(3):
        L.append({"event": "cut", "i": i})
    got = ST.Ledger(p).read()
    check("three lines, seq 0..2, each carrying the sha256 of the file before it",
          [e["seq"] for e in got] == [0, 1, 2] and got[0]["prev_sha256"] == hashlib.sha256(b"").hexdigest())
    data = p.read_bytes()
    lines = data.split(b"\n")
    lines[1] = lines[1].replace(b'"i": 1', b'"i": 7')
    assert lines[1] != data.split(b"\n")[1]
    p.write_bytes(b"\n".join(lines))
    check("an edited middle line breaks the chain: refused", raises(lambda: ST.Ledger(p).read(), ST.StreamError,
                                                                    "chain is broken"))
    p.write_bytes(data + b'{"event": "cu')
    L2 = ST.Ledger(p)
    check("a partial last line is ignored by a reader and cut back (fragment kept) by a writer",
          len(L2.read()) == 3 and p.read_bytes().endswith(b'"cu') and len(L2.read(repair=True)) == 3
          and p.read_bytes() == data and list(p.parent.glob("ledger.partial.*")))
    L3 = ST.Ledger(p)
    L3.read()
    with open(p, "ab") as fh:
        fh.write(b"")
    L4 = ST.Ledger(p)
    L4.read()
    L4.append({"event": "cut", "i": 3})
    check("an append from a stale reader is refused (another writer)",
          raises(lambda: L3.append({"event": "cut", "i": 4}), ST.StreamError, "another writer"))


def new_stream(sid, base, m, step1, guard, fb=None, clock=None, stage_b=None, milestone0=None, baseline=None,
               secondary=None, submitter=None):
    deps = ST.Deps(guard=guard, hasher=hashes, backend=fb or D.FakeBackend(), baseline_runner=baseline,
                   secondary_runner=secondary, submitter=submitter)
    st = ST.Stream(sid, deps=deps, clock=clock or Clock(), quiet=True)
    st.init(base=base, m=m, testing=True, step1_dir=str(step1.root), stage_b=stage_b, milestone0=milestone0)
    return st


def test_init(base, base_rows, guard):
    print("init")
    s1 = Step1World("s1_init")
    s1.commit()
    st = new_stream("initw", base, None, s1, guard)
    f = st.load()
    led = st.ledger.read()
    pool = f.current_pool()
    check("P_0 holds base_v2's bytes, content-addressed; M = ceil(0.10 x |base|)",
          pool["sha256"] == C.sha256_file(base) and pathlib.Path(pool["path"]).name.startswith("P_0.%s" % pool["sha256"][:16])
          and f.defn["M"] == -(-len(base_rows) // 10), (pool, f.defn["M"]))
    check("the ledger starts with init then the prospective record (M, K, truth policy, thresholds, recipe rule, "
          "gate block), before any build",
          [e["event"] for e in led[:2]] == ["init", "prospective"]
          and set(led[1]["record"]) >= {"M", "K_max", "truth_policy", "thresholds", "recipe_rule", "gate_block"}
          and led[1]["record_sha256"] == ST._sha_obj(led[1]["record"]), [e["event"] for e in led])
    check("a stream is defined once", raises(lambda: new_stream("initw", base, None, s1, guard), ST.StreamError,
                                             "already exists"))
    check("Stage B must start with r0 and add at most one survivor; freeze/LoRA refused (L-6)",
          raises(lambda: ST.Stream._check_stage_b("x1a,r0"), ST.StreamError)
          and raises(lambda: ST.Stream._check_stage_b("r0,freeze"), ST.StreamError, "L-6")
          and ST.Stream._check_stage_b("r0,x1b") == ["r0", "x1b"])
    check("state.json, consumed.jsonl-ready summary and quarantine file written",
          st.p.state.is_file() and st.p.summary.is_file() and st.p.quarantine.is_file())


def test_cut(base, base_rows, guard, dev_paths):
    print("the cut")
    s1 = Step1World("s1_cut")
    tray = {"trayA1": 6, "trayA2": 7, "trayA3": 5}
    pair = pattern(next_seed())
    s1.add("srcA", "b0001", [0, 0, 1] + [0] * 9, n=1, capture_group="trayA1", pat=pair)
    s1.add("srcA", "b0001", [0, 0, 1] + [0] * 9, n=1, capture_group="trayA1", pat=pair, noise=2)
    s1.add("srcA", "b0001", [0, 0, 1] + [0] * 9, n=tray["trayA1"] - 2, capture_group="trayA1")
    s1.add("srcA", "b0001", [0, 0, 1] + [0] * 9, n=tray["trayA2"], capture_group="trayA2", l1=1)
    s1.add("srcA", "b0002", [0, 0, 1] + [0] * 9, n=tray["trayA3"], capture_group="trayA3", l1=2)
    mixed = [1, 1, 0, 0, 0, 1] + [0] * 6
    s1.add("srcB", "b0002", mixed, n=8, capture_group="vidB1")
    s1.add("srcC", "b0003", [0] * 8 + [2] + [0] * 3, n=6, capture_group="potC")
    held = s1.add("srcB", "b0002", mixed, n=1, capture_group="vidB9", holds=("h6_scan",))[0]
    other = s1.add("srcB", "b0002", [0] * 12, n=1, capture_group="vidB9", other=3)[0]
    unev = s1.add("srcD", "b0002", mixed, n=3, capture_group="d", evidenced=False)
    refused = s1.add("srcB", "b0002", mixed, n=1, capture_group="vidB9")[0]
    bcopy = s1.add("srcB", "b0002", mixed, n=1, capture_group="vidB9", image_from=base_rows[3]["image"])[0]
    s1.commit()
    s1.event(event="refuse", key=refused["key"], reason="superseded:rejoin_v1", batch="x")
    st = new_stream("cutw", base, 10, s1, guard)
    st.load()
    plans, refusal, ana = st.cut_plan(2)
    p1 = plans[0]
    keys1 = [r["key"] for r in p1["rows"]]
    check("increment 1: exactly M = 10 images", len(p1["rows"]) == 10, len(p1["rows"]))
    plans_b, _r, _a = st.cut_plan(2)
    check("deterministic: the same queue and seeds give the same increments",
          [[r["key"] for r in p["rows"]] for p in plans] == [[r["key"] for r in p["rows"]] for p in plans_b])
    check("seeded by stable_int('<sid>/<segment>/<step>')",
          p1["seed_text"] == "cutw/001/1" and p1["seed"] == C.stable_int("cutw/001/1"), p1["seed_text"])
    pair_keys = [s1.rows[0]["key"], s1.rows[1]["key"]]
    all_keys = [r["key"] for p in plans for r in p["rows"]]
    check("a near-duplicate pair is one unit: both in the same increment or neither",
          (pair_keys[0] in keys1) == (pair_keys[1] in keys1) and all_keys.count(pair_keys[0]) <= 1)
    bad = {held["key"], other["key"], refused["key"], bcopy["key"]} | {r["key"] for r in unev}
    check("never cut: the held row, the OtherPlant-only row, the refused row, the unevidenced source, the base copy",
          not (bad & set(all_keys)), bad & set(all_keys))
    check("the reasons are counted: held, kind_other, refused_in_queue, not_evidenced, in_pool or near_pool",
          ana["reasons"].get("held") == 1 and ana["reasons"].get("kind_other") == 1
          and ana["reasons"].get("refused_in_queue") == 1 and ana["reasons"].get("not_evidenced") == 3
          and (ana["reasons"].get("in_pool", 0) + ana["reasons"].get("near_pool_or_in_flight", 0)) == 1, ana["reasons"])
    srcs = collections.Counter(r["source"] for r in p1["rows"])
    check("one source first: the oldest source with >= M (srcA) leads increment 1", p1["first_source"] == "srcA"
          and srcs["srcA"] >= 5, srcs)
    cap = p1["species_cap"]
    check("the 60 %% cap applies with two or more sources and is met by swapping (Purslane share %.2f)"
          % (cap.get("max_share") or 0), cap["applies"] and cap["met"] and cap["swaps"] >= 1
          and cap["max_share"] <= 0.6 + 1e-9 and srcs["srcA"] < 10, cap)
    split = p1["split_capture_groups_keys"]
    check("a capture group that does not fit whole is split and recorded (the first source's trayA2)",
          ("srcA", "trayA2") in {tuple(x) for x in split}, split)
    rest = [r["key"] for r in s1.rows if (r["source"], r["capture_group"]) == ("srcA", "trayA2")
            and r["key"] not in keys1]
    keys2 = {r["key"] for r in plans[1]["rows"]}
    check("the split capture group's remainder is cut first into the next increment",
          rest and set(rest) <= keys2, (rest, sorted(keys2)))
    sm = st.summary()
    q_el = sm["queue"]["eligible_images"]
    check("queue_summary: Q, K = min(4, Q // M), an exact-fill probe, the reasons and holds",
          q_el == len(ana["cuttable"]) == len(ana["eligible"]) and sm["cut"]["k"] == min(4, q_el // 10)
          and sm["cut"]["probe"]["exact_fill"]
          and sm["queue"]["held"] == {"h6_scan": 1} and sm["queue"]["held_past_deadline"] == {"h6_scan": 1},
          (q_el, sm["cut"], sm["queue"]["held"]))
    check("queue_summary 'held' is {hold: {rows, past_deadline}}, the shape the autopilot's DHOLD reads (S26)",
          sm["held"] == {"h6_scan": {"rows": 1, "past_deadline": 1}}, sm["held"])
    # the eligible target rows only
    check("every cut row holds >= 1 kept verified target box", all(sum(r["species"]) >= 1 for p in plans for r in p["rows"]))
    # the copy scan is never released by hand
    kf = TMP / "release_keys.txt"
    kf.write_text(held["key"] + "\n")
    n_led = len(st.ledger.read())
    check("h6_scan is never released by hand, even by a person (by key, by source or stream-wide): the copy scan "
          "is mandatory (D-C, P9)",
          raises(lambda: st.release("h6_scan", "scan done", "human:owner", keys_file=str(kf)), ST.StreamError,
                 "mandatory")
          and raises(lambda: st.release("h6_scan", "scan done", "human:owner", source="srcB"), ST.StreamError,
                     "mandatory")
          and raises(lambda: st.release("h6_scan", None, "human:owner"), ST.StreamError, "mandatory")
          and len(st.ledger.read()) == n_led)
    # a person's release lifts a licence hold
    lic = s1.add("srcB", "b0002", mixed, n=1, capture_group="vidB9", holds=("licence",))[0]
    s1.commit()
    kf.write_text(lic["key"] + "\n")
    check("releasing a hold needs a person", raises(lambda: st.release("licence", "licence resolved", "platform",
                                                                        keys_file=str(kf)), ST.StreamError, "person"))
    st.release("licence", "licence resolved (owner)", "human:owner", keys_file=str(kf))
    st.load()
    _p, _r, ana2 = st.cut_plan(1)
    el2 = {e["key"] for e in ana2["eligible"]}
    check("after the person's release the licence-held row is eligible; the h6_scan row stays held",
          lic["key"] in el2 and held["key"] not in el2)
    e_lic = next((e for e in ana2["eligible"] if e["key"] == lic["key"]), {})
    check("a keys-scoped licence release applies to the row it lifts, fail closed (research_only, the release "
          "named); a row of the same source it did not lift carries none",
          (e_lic.get("licence_release") or {}).get("by") == "human:owner"
          and e_lic["licence_release"].get("research_only") is True and ST._row_licence(e_lic)[1] is True
          and all(e.get("licence_release") is None for e in ana2["eligible"]
                  if e["source"] == "srcB" and e["key"] != lic["key"]),
          (e_lic.get("licence_release"), ST._row_licence(e_lic) if e_lic else None))
    check("a stream-wide release is for funnel_F9 only (a licence is released row by row or by source)",
          raises(lambda: st.release("licence", None, "human:owner"), ST.StreamError, "funnel_F9"))
    f9 = s1.add("srcB", "b0002", mixed, n=1, capture_group="vidB9", holds=("funnel_F9",))[0]
    s1.commit()
    os.environ["INCAP_DECIDED_BY"] = "human:owner@example.org"
    try:
        rc = ST.main(["release", "--stream", "cutw", "--hold", "funnel_F9"], deps=st.deps)
    finally:
        os.environ.pop("INCAP_DECIDED_BY")
    st.load()
    _p, _r, ana3 = st.cut_plan(1)
    rel = [e for e in st.ledger.read() if e["event"] == "release"][-1]
    check("the R3 form 'release --stream SID --hold funnel_F9' (the approver from INCAP_DECIDED_BY) releases every "
          "funnel_F9 row", rc == 0 and rel["scope"] == "all" and rel["by"] == "human:owner@example.org"
          and f9["key"] in {e["key"] for e in ana3["eligible"]}, (rc, rel))
    check("without a person's approval the same argv is refused",
          ST.main(["release", "--stream", "cutw", "--hold", "funnel_F9"], deps=st.deps) == 1)
    # single source: no cap
    s1b = Step1World("s1_single")
    s1b.add("solo", "b0001", [0, 0, 3] + [0] * 9, n=12, capture_group="t")
    s1b.commit()
    st2 = new_stream("singlew", base, 10, s1b, guard)
    st2.load()
    (ps,), _r, _a = st2.cut_plan(1)
    check("one source is all there is: the cap does not apply (one species, 100 %)",
          not ps["species_cap"]["applies"] and ps["species_cap"]["max_share"] == 1.0)


def test_test_v1_never_cut(base, guard):
    print("a row a main-test list holds (inc2.base3 splits/*/test_v1) is never cut")
    from weed_optimizer_framework.tools.inc2 import step1_stream as S1
    s1 = Step1World("s1_testv1")
    rows = s1.add("srcT", "b0001", [0, 0, 1] + [0] * 9, n=12, capture_group="trayT1")
    s1.commit()
    same, near = rows[0], rows[1]
    flip = int(near["dhash"]) ^ 0b11                       # 2 bits away
    tl = pathlib.Path(C.INC_DIR) / "splits" / "v3" / "test_v1" / "srcT.jsonl"
    tv = [dict(key="tv1_a", image="/x/a.png", label="/x/a.txt", sha256="0" * 64, label_sha256="0" * 64, source="srcT",
               session="g", original_sha256=same["unmasked_sha256"], dhash=None, variants=None),
          dict(key="tv1_b", image="/x/b.png", label="/x/b.txt", sha256="1" * 64, label_sha256="1" * 64, source="srcT",
               session="g", original_sha256="2" * 64, dhash=flip, variants=[flip] * 8)]
    try:
        tl.parent.mkdir(parents=True, exist_ok=True)       # as inc2.base3 writes it: provenance rows, not a manifest
        tl.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in tv))
        q = S1.load_queue(S1.Layout(root=s1.root, inc_dir=s1.root.parent))
        by = {r["key"]: r for r in q}
        check("step1_stream.load_queue marks the copy (bytes) and the near row (dHash 2 bits) test_v1, not eligible",
              by[same["key"]].get("test_v1") == "bytes" and by[near["key"]].get("test_v1") == "dhash_2"
              and not by[same["key"]]["eligible_step1"] and not by[near["key"]]["eligible_step1"]
              and sum(1 for r in q if r.get("test_v1")) == 2, [(k, r.get("test_v1")) for k, r in by.items()][:4])
        st = new_stream("testv1w", base, 4, s1, guard)
        st.load()
        plans, _refusal, ana = st.cut_plan(2)
        cut = {r["key"] for p in plans for r in p["rows"]}
        check("  the cutter never takes them (reason test_v1), and cuts the rest",
              ana["reasons"].get("test_v1") == 2 and not ({same["key"], near["key"]} & cut) and cut,
              (ana["reasons"], len(cut)))
        tl.write_text("{not json\n")
        try:
            S1.load_queue(S1.Layout(root=s1.root, inc_dir=s1.root.parent))
            got = None
        except Exception as e:  # noqa: BLE001
            got = e
        check("  an unreadable test list refuses the queue's read (fail closed)", got is not None, got)
    finally:
        if tl.exists():
            tl.unlink()


def test_licence_release(base, guard):
    print("a licence release fails closed: research_only unless the person records the licence and says not")
    M = 4
    mixed = [1, 1, 0, 0, 0, 1] + [0] * 6
    s1 = Step1World("s1_licrel")
    for r in s1.add("lic_a", "b0001", mixed, n=M, capture_group="la", holds=("licence",)):
        r["licence"] = None                           # a row held licence records none
    s1.commit()
    st = new_stream("licw", base, M, s1, guard)
    check("--not-research-only needs the person's licence text; --licence belongs to a licence release; nothing "
          "recorded", raises(lambda: st.release("licence", "r", "human:owner", source="lic_a", research_only=False),
                             ST.StreamError, "--licence TEXT")
          and raises(lambda: st.release("funnel_F9", "r", "human:owner", licence="CC BY 4.0"), ST.StreamError,
                     "licence release")
          and not [e for e in st.ledger.read() if e["event"] == "release"])
    check("--not-research-only is refused for a licence text that names no known licence or restricts use (read as "
          "step1_stream reads an override); nothing recorded",
          all(raises(lambda t=t: st.release("licence", "r", "human:owner", source="lic_a", licence=t,
                                            research_only=False), ST.StreamError, "stay research_only")
              for t in ("unknown", "Custom terms", "CC BY-NC 4.0", "CC BY 4.0, academic use only", "research-only"))
          and ST.main(["release", "--stream", "licw", "--hold", "licence", "--source", "lic_a", "--licence",
                       "unknown", "--not-research-only", "--decided-by", "human:owner"], deps=st.deps) == 1
          and not [e for e in st.ledger.read() if e["event"] == "release"])
    st.release("licence", "the owner accepts the source", "human:owner", source="lic_a")
    rel = [e for e in st.ledger.read() if e["event"] == "release"][-1]
    check("a licence release without the person's licence text records none, and research_only true",
          rel["licence"] is None and rel["research_only"] is True, rel)
    exp = st.build(1)
    f = st.load()
    inc = f.segments[1]["increments"][0]
    side = f.inc_rows(inc)
    meta = json.loads(st.p.inc_meta(inc).read_text())
    ro = json.loads(D.Paths(exp).exp_json.read_text())["stream"]["research_only"]
    check("its rows are research_only in the cut's rows sidecar and meta (fail closed), the release named; the "
          "segment's models are research_only",
          len(side) == M and all(r["research_only"] is True and r["licence"] is None
                                 and (r["licence_release"] or {}).get("by") == "human:owner" for r in side.values())
          and meta["research_only_rows"] == M and ro["models"] is True,
          ([(r["research_only"], r["licence"], r.get("licence_release")) for r in side.values()], meta.get(
              "research_only_rows"), ro))
    s2 = Step1World("s1_licrel2")
    for r in s2.add("lic_b", "b0001", mixed, n=M - 1, capture_group="lb", holds=("licence",)):
        r["licence"] = None
    tainted = s2.add("lic_b", "b0001", mixed, n=1, capture_group="lb", holds=("licence",), research_only=True)[0]
    tainted["licence"] = None
    s2.commit()
    st2 = new_stream("licw2", base, M, s2, guard)
    rc = ST.main(["release", "--stream", "licw2", "--hold", "licence", "--source", "lic_b", "--licence", "CC BY 4.0",
                  "--not-research-only", "--decided-by", "human:owner", "--reason", "the licence is on the record"],
                 deps=st2.deps)
    rel2 = [e for e in st2.ledger.read() if e["event"] == "release"][-1]
    check("the CLI records the person's licence text and research_only false",
          rc == 0 and rel2["licence"] == "CC BY 4.0" and rel2["research_only"] is False, (rc, rel2))
    st2.build(1)
    f2 = st2.load()
    side2 = f2.inc_rows(f2.segments[1]["increments"][0])
    check("with the licence text and --not-research-only the rows carry that licence and are not research_only; a "
          "row the queue marks research_only stays so",
          len(side2) == M and all(r["licence"] == "CC BY 4.0" for r in side2.values())
          and all(r["research_only"] is (k == tainted["key"]) for k, r in side2.items()),
          [(k, r["licence"], r["research_only"]) for k, r in side2.items()])


def test_guard_at_cut(base, base_rows, guard, dev_paths):
    print("never-train v2 at the cut (planted copies of a dev image)")
    s1 = Step1World("s1_guard")
    dev0 = pattern(4242)
    save_pattern(dev0, dev_paths[0])
    mixed = [1, 1, 0, 0, 0, 1] + [0] * 6
    exact = s1.add("srcP", "b0001", mixed, capture_group="p0", pat=dev0, noise=2)[0]
    near3 = s1.add("srcP", "b0001", mixed, capture_group="p1", pat=near_pattern(dev0, 2, seed=1))[0]
    six = None
    for sd in range(40):
        cand = near_pattern(dev0, 3, seed=100 + sd)
        tmpp = save_pattern(cand, TMP / "probe6.png")
        d = bin(hashes(tmpp)[0] ^ hashes(dev_paths[0])[0]).count("1")
        if 4 <= d <= 6:
            six = cand
            break
    near6 = s1.add("srcP", "b0001", mixed, capture_group="p2", pat=six)[0]
    hf = TMP / "hflip.png"
    Image.open(dev_paths[0]).transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(hf)
    hflip = s1.add("srcP", "b0001", mixed, capture_group="p3", image_from=hf)[0]
    rf = TMP / "rot90.png"
    Image.open(dev_paths[0]).rotate(90, expand=True).save(rf)
    rot = s1.add("srcP", "b0001", mixed, capture_group="p4", image_from=rf)[0]
    dev1 = pattern(4243)
    masked = s1.add("srcP", "b0001", mixed, capture_group="p5", admission="masked", unmasked_pattern=dev1)[0]
    clean = s1.add("srcP", "b0002", mixed, n=12, capture_group="c")
    # the planted rows are in the queue as step1_stream would never admit them; strip their hashes so only the
    # cut's own GuardV2 check can stop them
    s1.commit()
    st = new_stream("guardw", base, 10, s1, guard)
    st.load()
    (p,), refusal, ana = st.cut_plan(1)
    planted = {exact["key"], near3["key"], near6["key"], hflip["key"], rot["key"], masked["key"]}
    got = {r["key"] for r in p["rows"]}
    check("dHash of the 6-bit copy is 4..6 bits from the dev image", six is not None)
    check("no planted copy is cut: exact, 3-bit, %s-bit, hflip, rot90, and the masked row with a dev original"
          % "4-6", not (planted & got) and len(got) == 10 and got <= {r["key"] for r in clean}, planted & got)
    ge = ana["guard_excluded"]
    check("each is refused by GuardV2 and counted by reason, per unit: the exact, 3-bit and 6-bit copies are one "
          "6-bit unit (3 images), hflip and rot90 hit through a variant, the masked row through its original",
          ge.get("train:near_eval_v2", 0) == 3 and ge.get("train:near_eval_variant", 0) == 2
          and ge.get("unmasked:near_eval_v2", 0) == 1, ge)


def test_choose_arm(base, guard):
    print("L-4: the stream adopts inc2.baseline's capacity decision")
    s1 = Step1World("s1_arm")
    s1.commit()
    st = new_stream("armw", base, 5, s1, guard)
    cap = TMP / "capacity_v1.json"
    from weed_optimizer_framework.tools.inc2 import recipes as RC
    meas = RC.step_cost(7625, 763, arm="s640", recipe="r0", truth=True)["gpu_h"][1] * 0.5
    cap.write_text(json.dumps({"format": "inc2-capacity/1", "chosen_arm": "s640", "chosen_exp": "b_v2_s640",
                               "qualifying": ["b_v2_s640"], "n_images": 7625, "m": 763, "truth_every": 1,
                               "step_cost": {"gpu_h": [meas, meas], "estimate": False},
                               "arms": {"b_v2": {"arm": "n640", "mean": 0.80, "sd": 0.004},
                                        "b_v2_s640": {"arm": "s640", "mean": 0.83, "sd": 0.004, "qualifies": True}},
                               "chosen_arm_record": {"id": "s640"}}))
    rec = st.choose_arm(cap)
    f = st.load()
    ev = [e for e in st.ledger.read() if e["event"] == "arm"][-1]
    check("the chosen arm, milestone 0 = its R0 baseline, the measured cost ratio, the decision file by sha256",
          rec["id"] == "s640" and abs(rec["cost_multiplier"] - 0.5) < 1e-9 and f.milestones[0]["exp"] == "b_v2_s640"
          and ev["capacity"]["sha256"] == C.sha256_file(cap) and ev["arms"]["s640"]["exp"] == "b_v2_s640", rec)
    bad = TMP / "bad_capacity.json"
    bad.write_text(json.dumps({"format": "x"}))
    check("anything but inc2.baseline's capacity decision is refused", raises(lambda: st.choose_arm(bad), ST.StreamError))
    n_arm = len([e for e in st.ledger.read() if e["event"] == "arm"])
    for aid in RC.MEASURE_ARMS:
        meas_cap = TMP / ("capacity_%s.json" % aid)
        meas_cap.write_text(json.dumps(dict(json.loads(cap.read_text()), chosen_arm=aid, chosen_exp="b_v2_%s" % aid,
                                            qualifying=["b_v2_%s" % aid])))
        check("a decision naming the measurement arm %s is refused: never a stream's arm" % aid,
              raises(lambda: st.choose_arm(meas_cap), ST.StreamError, contains="measurement arm"))
    check("... and the stream's arm is still the one the decision chose (s640), with no new arm line",
          (st.load().arm or {}).get("id") == "s640"
          and len([e for e in st.ledger.read() if e["event"] == "arm"]) == n_arm, st.load().arm)
    deps = ST.Deps(guard=guard, hasher=hashes, backend=D.FakeBackend())
    check("a stream is never created on a measurement arm (init --arm m832)",
          raises(lambda: ST.Stream("armw_m832", deps=deps, clock=Clock(), quiet=True).init(
              base=base, m=5, testing=True, step1_dir=str(s1.root), arm="m832"), ST.StreamError,
              contains="measurement arm"))


def test_deadlock(base, guard):
    print("S24: capture groups that cannot sum to M, units that cannot")
    s1 = Step1World("s1_dead")
    mixed = [1, 1, 0, 0, 0, 1] + [0] * 6
    for cg in ("big1", "big2", "big3"):
        s1.add("srcM", "b0001", mixed, n=7, capture_group=cg)
    s1.commit()
    st = new_stream("deadw", base, 10, s1, guard)
    st.load()
    (p,), refusal, _a = st.cut_plan(1)
    check("capture groups of 7, 7, 7 fill M = 10 exactly by splitting one by near-dup unit",
          len(p["rows"]) == 10 and refusal is None and p["split_capture_groups_keys"], p["split_capture_groups_keys"])
    s2 = Step1World("s1_dead2")
    for g in range(3):
        a = pattern(next_seed())
        for j in range(7):
            s2.add("srcN", "b0001", mixed, capture_group="g%d" % g, pat=a, noise=2)
    s2.commit()
    st2 = new_stream("dead2w", base, 10, s2, guard)
    st2.load()
    plans, refusal, ana = st2.cut_plan(1)
    check("units of 7 near-identical frames cannot make 10: no increment, a refusal with the reason",
          not plans and refusal and refusal["why"] == "no_exact_fill" and refusal["eligible_images"] == 21,
          refusal)
    rc = ST.main(["build", "--stream", "dead2w", "--k", "1"], deps=st2.deps)
    ref = json.loads(st2.p.cut_refusal.read_text())
    check("build refuses (exit 1) and writes cut_refusal.json for D22, never a silent wait",
          rc == 1 and ref["refusal"]["why"] == "no_exact_fill" and not st2.load().segments, ref)


def chain_patterns(n, seed0):
    """n patterns whose dHashes chain: each within 4..6 bits of the one before and > 3 bits from every earlier
    one (one transitive 6-bit group of n images; every 3-bit near-duplicate group a single image)."""
    probe = TMP / ("chain_probe_%d.png" % seed0)
    out, hs = [pattern(seed0)], [hashes(save_pattern(pattern(seed0), probe))[0]]
    tries = 0
    while len(out) < n:
        tries += 1
        if tries > 20000:
            raise RuntimeError("no chain found")
        cand = near_pattern(out[-1], 2, seed=seed0 * 100000 + tries)
        hc = hashes(save_pattern(cand, probe))[0]
        if 4 <= bin(hc ^ hs[-1]).count("1") <= 6 and all(bin(hc ^ x).count("1") > 3 for x in hs):
            out.append(cand)
            hs.append(hc)
    return out


def test_split_units(base, guard):
    print("§3.4 [review] / S24: a near-duplicate chain longer than M is split by 3-bit group, never idles")
    mixed = [1, 1, 0, 0, 0, 1] + [0] * 6
    s1 = Step1World("s1_chain")
    chain = [s1.add("vidS", "b0001", mixed, capture_group="video1", pat=a)[0] for a in chain_patterns(12, 77)]
    s1.add("vidS", "b0002", mixed, n=10, capture_group="video2")
    s1.commit()
    st = new_stream("chainw", base, 10, s1, guard)
    st.load()
    plans, refusal, ana = st.cut_plan(2)
    ck = {r["key"] for r in chain}
    k1 = [r["key"] for r in plans[0]["rows"]] if plans else []
    k2 = [r["key"] for r in plans[1]["rows"]] if len(plans) > 1 else []
    check("a 6-bit chain of 12 frames (> M = 10) is split by 3-bit group: increment 1 takes 10 of its frames, "
          "the split recorded", len(k1) == 10 and set(k1) <= ck and plans[0]["split_near_dup_groups"]
          and plans[0]["split_near_dup_groups"][0]["parent_size"] == 12
          and plans[0]["split_near_dup_groups"][0]["taken"] == 10, (k1, plans[0].get("split_near_dup_groups")
                                                                    if plans else refusal))
    check("the chain's other 2 frames wait for a later cut: increment 2 of the same cut is the 10 independent "
          "images (no two near-copies in different increments of one pool)",
          len(k2) == 10 and not (set(k2) & ck), (k2, refusal))
    plans_ng, _r_ng, _a_ng = st.cut_plan(2, guard=False)
    k2ng = [r["key"] for r in plans_ng[1]["rows"]] if len(plans_ng) > 1 else []
    check("the separation holds by the unit rule alone (without the cut's variant check): a split group's other "
          "units never go to another increment of the same cut", len(k2ng) == 10 and not (set(k2ng) & ck), k2ng)
    sm = st.summary()
    check("queue_summary counts the split chain as supply (22 cuttable images, nothing uncuttable)",
          sm["eligible"]["images"] == 22 and sm["eligible"]["uncuttable"]["images"] == 0, sm["eligible"])
    s2 = Step1World("s1_blob")
    a = pattern(next_seed())
    blob = [s2.add("vidT", "b0001", mixed, capture_group="t", pat=a, noise=2)[0] for _ in range(11)]
    s2.add("vidT", "b0002", mixed, n=3, capture_group="u")
    s2.commit()
    st2 = new_stream("blobw", base, 10, s2, guard)
    st2.load()
    sm2 = st2.summary()
    plans2, refusal2, _a2 = st2.cut_plan(1)
    check("11 frames within 3 bits of one another (one near-duplicate group > M) are uncuttable: reported, not "
          "counted as supply (Q = 3, so D20 collects), and the cut refuses with the reason",
          sm2["eligible"]["images"] == 3 and sm2["eligible"]["uncuttable"] == {"images": 11, "units_larger_than_M": 1,
                                                                             "mixed_verifier_units": 0}
          and not plans2 and refusal2 and refusal2.get("units_larger_than_M") == 1
          and refusal2.get("images_in_units_larger_than_M") == 11, (sm2["eligible"], refusal2))
    del blob


class SecondaryDouble:
    def __init__(self):
        self.calls = []

    def __call__(self, exp, weights, source):
        self.calls.append((exp, weights, source))
        try:
            from weed_optimizer_framework.tools.inc2 import baseline as B
            res = B.secondary(exp, weights, source=source)
            return res["argv"]
        except Exception as e:  # noqa: BLE001 - group B's secondary unavailable: an equivalent spec
            print("       NOTE: inc2.baseline.secondary not used (%s: %s)" % (type(e).__name__, e))
            out = D.Paths(exp).run_dir(ST.SECONDARY_RUN)
            spec = {"exp": exp, "run_id": ST.SECONDARY_RUN, "kind": "final", "init": str(weights),
                    "exams": ["dev", "imageweeds", "test"], "out_dir": str(out)}
            D._write_json(out / "spec.json", spec)
            lst = D.Paths(exp).root / "submissions" / "secondary.txt"
            lst.parent.mkdir(parents=True, exist_ok=True)
            lst.write_text("%s\n" % (out / "spec.json"))
            return ["sbatch", "--parsable", "--array=0-0", "run_inc2_job.sh", str(lst), exp]


class SubmitDouble:
    """Runs the one spec of a secondary submission through the executor."""

    def __init__(self, executor):
        self.ex = executor
        self.argvs = []

    def __call__(self, argv):
        self.argvs.append(list(argv))
        lst = pathlib.Path(argv[-2])
        spec = json.loads(pathlib.Path(lst.read_text().strip()).read_text())
        self.ex(spec)
        return "777"


def test_end_to_end(base, base_rows, guard):
    print("end to end: segments on FakeBackend, commit, milestones, rollback, bisect")
    M = 6
    ex = Executor(M)
    fb = D.FakeBackend()
    clock = Clock()
    # milestone 0: a 5-seed baseline on P_0 (the R0 baseline, here built directly)
    bdl = BaselineDouble(fb)
    bdl(["build", "--exp", "e2e_m0", "--manifest", str(base), "--seeds", "0,1,2,3,4", "--arm", "n640",
         "--role", "b_v2", "--final-exams", "dev,imageweeds,test", "--testing"])
    drive("e2e_m0", fb, ex)
    s1 = Step1World("s1_e2e")
    mixed = [1, 1, 0, 0, 0, 1] + [0] * 6
    s1.add("good_a", "b0001", mixed, n=M, capture_group="g")
    bad_rows = s1.add("bad_a", "b0002", mixed, n=M, capture_group="b")
    s1.commit()
    sec, sub = SecondaryDouble(), SubmitDouble(ex)
    st = new_stream("e2e", base, M, s1, guard, fb=fb, clock=clock, milestone0="e2e_m0", baseline=bdl,
                    secondary=sec, submitter=sub)
    # ---- segment 1
    exp1 = st.build(2)
    defn = json.loads(D.Paths(exp1).exp_json.read_text())
    ok_valid = True
    try:
        D.validate_definition(defn)
        D.check_definition_data(defn)
    except D.DriverError as e:
        ok_valid = str(e)
    check("L18 built %s; exp.json passes the pinned validate_definition and check_definition_data" % exp1,
          exp1 == "e2e_s001" and ok_valid is True, ok_valid)
    check("the definition: chain, replay full, gate net, seeds 0-2, finals dev + imageweeds, clean false steps, "
          "base = P_0's content-addressed file, truth on, the arm stamp (protocol v3, inc2)",
          defn["type"] == "chain" and defn["replay_mode"] == "full" and defn["gate"] == {"flips_mode": "net"}
          and defn["seeds"] == [0, 1, 2] and defn["final_exams"] == ["dev", "imageweeds"]
          and all(s["clean"] is False for s in defn["steps"]) and defn["base"]["name"] == "P_0"
          and defn["base"]["manifest_sha256"] == C.sha256_file(base) and defn["truth"] is True
          and defn.get("protocol") == "v3" and defn.get("protocol_package") == "inc2"
          and defn.get("arm", {}).get("id") == "n640" and defn.get("init_weights") == "yolo11n.pt",
          {k: defn.get(k) for k in ("replay_mode", "gate", "final_exams", "truth", "protocol", "arm")})
    steps = [s["name"] for s in defn["steps"]]
    inc_rows = {s: C.read_manifest(ST.StreamPaths("e2e").inc_manifest(s)) for s in steps}
    check("two increments of exactly M, one source each (oldest first: good, then bad)",
          [len(v) for v in inc_rows.values()] == [M, M]
          and {kind_of_key(r["key"]) for r in inc_rows[steps[0]]} == {"good"}
          and {kind_of_key(r["key"]) for r in inc_rows[steps[1]]} == {"bad"}, {k: len(v) for k, v in inc_rows.items()})
    check("the build's runs went out as one array on the FakeBackend", len(fb.submissions) >= 2)
    summ = json.loads(st.p.summary.read_text())
    check("queue_summary: the segment is in flight, the TRAIN lane busy",
          summ["in_flight"] == [exp1] and summ["cut"]["train_idle"] is False)
    check("commit of an unfinished segment is refused", raises(lambda: st.commit(exp1), ST.StreamError, "not finished"))
    n_led = len(st.ledger.read())
    check("a second build while a segment is in flight is refused before any cut (L18 <= 1 in flight)",
          raises(lambda: st.build(1), ST.StreamError, "in flight") and len(st.ledger.read()) == n_led
          and ST.main(["build", "--stream", "e2e", "--k", "1"], deps=st.deps) == 1)
    drive(exp1, fb, ex)
    summ = st.write_summary()
    check("D23's input: the finished segment is listed as uncommitted", summ["uncommitted_done"] == [exp1],
          summ["uncommitted_done"])
    saved = ST._inc2
    ST._inc2 = lambda name: None if name == "gate3" else saved(name)
    try:
        check("commit without inc2.gate3 (Protocol v3) refuses: fail closed",
              raises(lambda: ST.Stream("e2e", deps=st.deps, quiet=True).commit(exp1), ST.StreamError, "gate3"))
    finally:
        ST._inc2 = saved
    res = st.commit(exp1)
    f = st.load()
    p1 = f.current_pool()
    p1_rows = C.read_manifest(p1["path"])
    check("S5: ACCEPT good, REJECT bad as data", res["dispositions"] == {steps[0]: "accepted", steps[1]: "data"},
          res["dispositions"])
    check("P_1 = P_0 + the accepted increment (content-addressed, pairwise disjoint)",
          p1["name"] == "P_1" and len(p1_rows) == len(base_rows) + M
          and {r["key"] for r in p1_rows} == {r["key"] for r in base_rows} | {r["key"] for r in inc_rows[steps[0]]}
          and pathlib.Path(p1["path"]).name == "P_1.%s.jsonl" % C.sha256_file(p1["path"])[:16], p1)
    cons = [json.loads(ln) for ln in st.p.consumed.read_text().splitlines()]
    q = json.loads(st.p.quarantine.read_text())
    check("consumed.jsonl records every key's disposition; the data-blamed images' dHashes are quarantined",
          collections.Counter(c["disposition"] for c in cons) == {"accepted": M, "data": M}
          and {e["key"] for e in q["entries"]} == {r["key"] for r in inc_rows[steps[1]]}, collections.Counter(
              c["disposition"] for c in cons))
    seg_rep = json.loads((D.Paths(exp1).root / "report.json").read_text())
    check("the segment report is written with the segment's exams and the stream commit",
          seg_rep["exams"] == ["dev", "imageweeds"] and seg_rep["stream_commit"]["chosen"] == "r0"
          and (D.Paths(exp1).root / "report.md").is_file())
    srep = json.loads(st.p.report_json.read_text()) if st.p.report_json.is_file() else {}
    check("the commit refreshes the stream report (§3.7): its timeline holds both increments with their "
          "dispositions", [t.get("disposition") for t in srep.get("timeline", [])] == ["accepted", "data"]
          and st.p.report_md.is_file(), srep.get("timeline"))
    # a re-harvest of a quarantined photograph under a new key
    s1.add("bad_b", "b0003", mixed, n=1, capture_group="rh", image_from=bad_rows[0]["image"])
    s1.add("bad_b", "b0003", mixed, n=1, capture_group="rh", pat=None)
    s1.commit()
    _p, _r, ana = st.cut_plan(1)
    check("a re-harvested copy of a quarantined image is refused by dHash (quarantined_dhash)",
          ana["reasons"].get("quarantined_dhash") == 1, ana["reasons"])
    # a mirrored re-upload of a quarantined image and a rotated one of an accepted image: other id dHashes (they
    # pass eligibility), the same photographs (their flips / rotations match)
    hfq, r9a = TMP / "e2e_hflip_quarantined.png", TMP / "e2e_rot90_accepted.png"
    Image.open(bad_rows[1]["image"]).transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(hfq)
    Image.open(inc_rows[steps[0]][0]["image"]).rotate(90, expand=True).save(r9a)
    mir = (s1.add("mirror_a", "b0003", mixed, n=1, capture_group="mm", image_from=hfq)
           + s1.add("mirror_a", "b0003", mixed, n=1, capture_group="mm", image_from=r9a)
           + s1.add("mirror_a", "b0003", mixed, n=M - 2, capture_group="mm"))
    s1.commit()
    _p, refm, anam = st.cut_plan(1)
    gem = anam["guard_excluded"]
    check("a mirrored re-upload of a quarantined image and a rotated one of an accepted image pass the id-dHash "
          "eligibility but are refused at the cut through their variants",
          {mir[0]["key"], mir[1]["key"]} <= {e["key"] for e in anam["eligible"]}
          and gem.get("train:" + ST.VARIANT_QUARANTINE) == 1 and gem.get("train:" + ST.VARIANT_POOL) == 1
          and not any(r["key"] in (mir[0]["key"], mir[1]["key"]) for p in _p for r in p["rows"]), (gem, refm))
    for r in mir:                                   # out of the rest of this world
        s1.event(event="refuse", key=r["key"], reason="test: mirrored re-upload case done", batch="x")
    # ---- milestone 1: build, run, compare (helps)
    st.milestone()
    m1 = ST.milestone_exp("e2e", 1)
    check("L20 calls inc2.baseline build with 5 seeds on the current pool, role milestone, the arm and 3 exams",
          bdl.argvs[-1][:3] == ["build", "--exp", m1] and "--seeds" in bdl.argvs[-1]
          and bdl.argvs[-1][bdl.argvs[-1].index("--seeds") + 1] == "0,1,2,3,4"
          and bdl.argvs[-1][bdl.argvs[-1].index("--manifest") + 1] == p1["path"]
          and "--role" in bdl.argvs[-1] and "milestone" in bdl.argvs[-1], bdl.argvs[-1])
    inc_rec = json.loads((st.p.milestone_dir(1) / "incumbent.json").read_text())
    check("the chain incumbent's secondary scoring was written by inc2.baseline secondary and submitted",
          inc_rec["status"] == "submitted" and inc_rec["job_id"] == "777" and sec.calls
          and D.Paths(m1).score(ST.SECONDARY_RUN, "test").is_file(), inc_rec)
    drive(m1, fb, ex)
    cmp1 = st.milestone()
    check("milestone 1 vs milestone 0 on dev, 5 v 5: helps (no rollback)",
          cmp1["verdict"] == "helps" and not cmp1["rollback_recommended"] and cmp1["compared_with"] == "e2e_m0",
          cmp1)
    srep = json.loads(st.p.report_json.read_text())
    check("the milestone comparison refreshes the stream report: it leads with milestone 1's test and the gap",
          (srep.get("headline") or {}).get("milestone") == m1
          and (srep.get("headline") or {}).get("gap_to_target") is not None, srep.get("headline"))
    entry = (st.p.milestone_dir(1) / "research_log_entry.md").read_text()
    check("the milestone's RESEARCH_LOG entry is written (what changed, why, how verified, the test and gap)",
          all(x in entry for x in ("**What changed.**", "**Why.**", "**How it was verified.**", "gap to 0.90",
                                   "Sealed test mAP50-95")), entry)
    check("a milestone on an unchanged pool is refused", raises(lambda: st.milestone(), ST.StreamError, "nothing"))
    # ---- segment 2: flat (HOLD), recipe, bad x2
    for src in ("flat_a", "recipe_a", "bad_c", "bad_d"):
        s1.add(src, "b0004" if src in ("flat_a", "recipe_a") else "b0005", mixed, n=M, capture_group=src)
    s1.commit()
    exp2 = st.build(4)
    drive(exp2, fb, ex)
    res2 = st.commit(exp2)
    kinds2 = {i: kind_of_key(C.read_manifest(ST.StreamPaths("e2e").inc_manifest(i))[0]["key"])
              for i in res2["dispositions"]}
    by_kind = {kinds2[i]: d for i, d in res2["dispositions"].items()}
    check("segment 2: flat -> hold, recipe-only REJECT -> recipe, bad -> data; D30 silent (1 of 3 REJECTs)",
          by_kind.get("flat") == "hold" and by_kind.get("recipe") == "recipe" and by_kind.get("bad") == "data"
          and res2["d30"]["fires"] is False, (by_kind, res2["d30"]))
    f = st.load()
    flat_keys = [k for k, v in f.keys.items() if kind_of_key(k) == "flat"]
    check("the HOLD and recipe images returned once (counted)",
          all(f.keys[k]["status"] == "returned" and f.keys[k]["returns"] == 1 for k in flat_keys), flat_keys)
    # ---- segment 3: the returned images first, plus two bad sources
    for src in ("bad_e", "bad_f"):
        s1.add(src, "b0006", mixed, n=M, capture_group=src)
    s1.commit()
    exp3 = st.build(4)
    d3 = json.loads(D.Paths(exp3).exp_json.read_text())
    first_kinds = [kind_of_key(C.read_manifest(ST.StreamPaths("e2e").inc_manifest(s["name"]))[0]["key"])
                   for s in d3["steps"]]
    check("the returned (oldest) images are cut first", first_kinds[:2] == ["flat", "recipe"], first_kinds)
    drive(exp3, fb, ex)
    st.commit(exp3)
    f = st.load()
    recipe_keys = [k for k, v in f.keys.items() if kind_of_key(k) == "recipe"]
    check("a second counted non-accept makes them neutral (quarantined): the HOLD images and the recipe images",
          all(f.keys[k]["status"] == "neutral" for k in flat_keys + recipe_keys) and recipe_keys
          and {e["key"] for e in f.quarantine if e["reason"] == "neutral"} >= set(flat_keys + recipe_keys))
    # ---- segment 4: sneaky (accepted, hurts cold), good2, rare (species-only pinned REJECT; v3 ACCEPT)
    for src in ("sneaky_a", "good_b", "rare_a"):
        s1.add(src, "b0007", mixed, n=M, capture_group=src, research_only=src == "sneaky_a")
    s1.commit()
    exp4 = st.build(3)
    drive(exp4, fb, ex)
    res4 = st.commit(exp4)
    kinds4 = {kind_of_key(C.read_manifest(ST.StreamPaths("e2e").inc_manifest(i))[0]["key"]): i
              for i in res4["dispositions"]}
    seg4 = st.load().segments[4]["commit"]
    rare_step = next(s for s in seg4["steps"]["r0"] if s["increment"] == kinds4["rare"])
    check("Protocol v3 (inc2.gate3): the rare-species increment the pinned gate REJECTs on PricklySida alone is "
          "ACCEPTed (SE-based tolerance) and joins P_4; recorded as discordant",
          rare_step["pinned"]["verdict"] == "REJECT" and rare_step["pinned"]["species_failed"] == [RARE_SPECIES]
          and rare_step["v3"]["verdict"] == "ACCEPT" and rare_step["v3"]["v3_applied"]
          and res4["dispositions"][kinds4["rare"]] == "accepted"
          and any(x["increment"] == kinds4["rare"] for x in seg4["discordant"]), rare_step)
    g3 = D.Paths(exp4).root / "gate3.json"
    cev = [e for e in st.ledger.read() if e["event"] == "commit" and e["exp"] == exp4][0]
    check("the commit records inc2.gate3's document by sha256 (INC_DIR/<exp>/gate3.json)",
          g3.is_file() and cev["gate"]["gate3"]["sha256"] == C.sha256_file(g3) and cev["gate"]["v3_unavailable"] == [],
          cev["gate"])
    # ---- milestone 2: hurts -> rollback -> bisect
    st.milestone()
    m2 = ST.milestone_exp("e2e", 2)
    drive(m2, fb, ex)
    cmp2 = st.milestone()
    check("milestone 2 vs milestone 1 (the last good): 5 v 5 permutation p <= 0.025, lower mean -> hurts, "
          "rollback recommended to milestone 1's pool",
          cmp2["verdict"] == "hurts" and cmp2["perm_p"] <= 0.025 and cmp2["rollback_recommended"]
          and cmp2["to_pool"] == "P_1", cmp2)
    summ = st.write_summary()
    check("queue_summary lists the pending rollback (D25 -> L21)", summ["rollback_pending"] == [{"milestone": m2,
                                                                                                 "to_pool": "P_1"}])
    check("a rollback elsewhere than the recommended pool needs a person",
          raises(lambda: st.rollback("P_0"), ST.StreamError, "person"))
    suspect = st.rollback("P_1")
    f = st.load()
    check("rollback: the pool pointer is P_1 again; the increments accepted since are suspect and excluded",
          f.pool == "P_1" and set(suspect) == {kinds4["sneaky"], kinds4["good"], kinds4["rare"]}
          and all(f.increments[i]["status"] == "suspect" for i in suspect), (f.pool, suspect))
    check("one rollback per milestone: a second one needs a person", raises(lambda: st.rollback("P_1"), ST.StreamError))
    arms = st.bisect("P_1")
    for e in arms.values():
        drive(e, fb, ex)
    dec = st.bisect("P_1")
    f = st.load()
    check("bisect (L27): one cold 3-seed arm per suspect increment, compared with milestone 1's seeds: sneaky "
          "hurts -> quarantined as data; good helps -> returned to the queue",
          dec["decided"].get(kinds4["sneaky"]) == "hurts" and dec["decided"].get(kinds4["good"]) == "helps"
          and all(f.keys[k]["status"] == "quarantined" for k in f.inc_rows(kinds4["sneaky"]))
          and all(f.keys[k]["status"] == "returned_bisect" for k in f.inc_rows(kinds4["good"])), dec)
    bdef = json.loads(D.Paths(arms[kinds4["good"]]).exp_json.read_text())
    check("a bisect arm never reads test (finals dev only)", bdef["final_exams"] == ["dev"] and bdef["seeds"] == [0, 1, 2])
    bro = {k: json.loads(D.Paths(arms[kinds4[k]]).exp_json.read_text())["stream"].get("research_only")
           for k in ("sneaky", "good")}
    p1_ro = f.pools["P_1"]["research_only"]
    check("§8: a bisect arm records its models' research-only flag (P_c's plus the increment's rows): the arm with "
          "research_only rows is research-only, the other one carries P_c's flag",
          (bro["sneaky"] or {}).get("models") is True and bro["sneaky"]["increments_research_only_rows"] == M
          and (bro["good"] or {}).get("increments_research_only_rows") == 0 and bro["good"]["pool"] == p1_ro
          and bro["good"]["models"] == (True if p1_ro is True else ("unknown" if p1_ro == "unknown" else False)),
          (bro, p1_ro))
    import re
    check("every experiment the stream builds is named <sid>_[smcb]NNN (the autopilot's child_exp rule)",
          all(re.match(r"^e2e_[smcb][0-9]{3}$", e) for e in list(arms.values()) + [exp1, m1, m2]), sorted(arms.values()))
    summ = st.write_summary()
    check("X4 stays raised only for what the bisection did not separate", summ["x4"]["raised"] is False, summ["x4"])
    # ---- segment 5 by the CLI (the L18 / L19 argv): the bisect-returned increment and a recipe-only REJECT alone
    s1.add("recipe_b", "b0008", mixed, n=M, capture_group="rb")
    s1.commit()
    check("a build whose --exp is not the next segment is refused",
          ST.main(["build", "--stream", "e2e", "--k", "2", "--exp", "e2e_s009"], deps=st.deps) == 1)
    n_led = len(st.ledger.read())
    check("L18's --arch/--imgsz must name the stream's arm (n640): another arm is refused before any cut",
          ST.main(["build", "--stream", "e2e", "--k", "2", "--arch", "yolo11s", "--imgsz", "640"], deps=st.deps) == 1
          and ST.main(["build", "--stream", "e2e", "--k", "2", "--imgsz", "1024"], deps=st.deps) == 1
          and len(st.ledger.read()) == n_led)
    rc = ST.main(["build", "--stream", "e2e", "--k", "2", "--exp", "e2e_s005", "--arch", "yolo11n", "--imgsz", "640",
                  "--truth-every", "3"], deps=st.deps)
    f = st.load()
    incs5 = f.segments[5]["increments"] if 5 in f.segments else []
    tp5 = f.segments[5]["truth_policy"] if 5 in f.segments else {}
    check("the CLI reads back the autopilot's L18 argv (build --stream SID --k K --exp SID_sNNN --arch --imgsz "
          "--truth-every) and builds on the rolled-back pool P_1; the bisect-returned increment comes back as a new "
          "cut; --truth-every 3 never switches off a truth arm the stream's own policy keeps (every 1)",
          rc == 0 and f.segments[5]["base_pool"] == "P_1" and len(incs5) == 2
          and tp5.get("every") == 1 and tp5.get("every_requested") == 3 and tp5.get("on") is True
          and {kind_of_key(k) for k in f.inc_rows(incs5[0])} == {"good"}
          and set(f.inc_rows(incs5[0])) == set(f.inc_rows(kinds4["good"])), (rc, incs5))
    drive("e2e_s005", fb, ex)
    rc = ST.main(["commit", "--exp", "e2e_s005"], deps=st.deps)
    f = st.load()
    c5 = f.segments[5]["commit"]
    rb_keys = list(f.inc_rows(incs5[1]))
    check("L19 by the CLI; D30 fires (the one REJECT is recipe-only): that return is not counted",
          rc == 0 and c5["d30"]["fires"] and c5["dispositions"][incs5[1]] == "recipe"
          and all(f.keys[k]["status"] == "returned" and f.keys[k]["returns"] == 0 for k in rb_keys)
          and c5["dispositions"][incs5[0]] == "accepted" and f.pool == "P_5", c5["d30"])
    # ---- the boundary check (segments on different pools)
    b = summ["boundary_check"]
    check("the boundary check compares the newest segment's base seeds with the previous one's (3 v 3, dev)",
          b is not None and set(b) >= {"segment", "previous", "new_mean", "old_mean", "old_sd", "fires"}, b)
    return st, fb, ex


def test_test_blindness(st):
    print("test blindness (S12) and the queue_summary schema")
    s0 = st.summary()
    PERTURB.update({"test": 0.2, "imageweeds": -0.3})
    try:
        for exp_dir in pathlib.Path(C.INC_DIR).iterdir():
            runs = exp_dir / "runs"
            if not runs.is_dir():
                continue
            for rd in runs.iterdir():
                for exam in ("test", "imageweeds"):
                    p = rd / "scores" / ("%s.json" % exam)
                    if p.is_file():
                        s = json.loads(p.read_text())
                        s["map50_95"] += PERTURB[exam]
                        s["species_map50_95"] += PERTURB[exam]
                        p.write_text(json.dumps(s))
        s1 = st.summary()
    finally:
        PERTURB.clear()
    strip = lambda s: {k: v for k, v in s.items() if k != "generated_utc"}  # noqa: E731
    check("every test and ImageWeeds score perturbed: queue_summary is unchanged", strip(s0) == strip(s1))
    led = st.p.ledger.read_text()
    check("no test or ImageWeeds value in the ledger (no 'test' exam key, no imageweeds number)",
          '"test_mean' not in led and "imageweeds_mean" not in led and '"exam": "test"' not in led)
    need = {"format", "sid", "M", "K_max", "pool", "queue", "cut", "segments", "in_flight", "uncommitted_done",
            "next_segment", "milestones", "boundary_check", "rollback_pending", "x4", "dispositions",
            "species_deficit", "consumed_target_boxes_last2", "data_blamed_sources", "quarantined_sources",
            "ledger", "last_segment", "m_over_pool", "feasibility", "arm", "chosen_recipe",
            "eligible", "held", "consumed_last"}
    check("queue_summary.json schema (%s)" % ST.SUMMARY_FORMAT, need <= set(s0) and s0["format"] == ST.SUMMARY_FORMAT
          and {"ready", "k", "probe", "train_idle", "last_refusal"} <= set(s0["cut"])
          and {"eligible_images", "eligible_target_boxes", "held", "held_past_deadline", "not_eligible",
               "oldest_eligible_utc"} <= set(s0["queue"])
          and {"due", "due_reasons", "accepted_since", "segments_since", "in_flight", "last_good"} <= set(s0["milestones"])
          and {"images", "target_boxes", "oldest_utc"} <= set(s0["eligible"])
          and s0["eligible"]["images"] == s0["queue"]["eligible_images"]
          and {"current", "images", "sha256"} <= set(s0["pool"]), sorted(need - set(s0)))
    led = st.ledger.read()
    com = [e for e in led if e["event"] == "commit"]
    rb = [e for e in led if e["event"] == "rollback"][-1]
    bis = [e for e in led if e["event"] == "bisect"]
    check("the ledger's commit lines name the accepted increments and the recipe; bisect lines the rollback's utc "
          "(what the autopilot's D24 / DBIS read)",
          all(isinstance(e["accepted"], list) and e["recipe"] == e["chosen"] for e in com)
          and bis and all(e["rollback_utc"] == rb["utc"] for e in bis))


def test_lease_and_cli(st):
    print("the single-writer lease and the CLI")
    lease = st.p.lease
    lease.write_text(json.dumps({"token": "other", "host": __import__("socket").gethostname(), "pid": os.getpid(),
                                 "expires_ts": 4e9}))
    check("a writing verb while another writer holds stream.lease: LeaseBusy, nothing appended",
          raises(lambda: st.withdraw("e2e_s999", "x"), ST.LeaseBusy))
    rc = ST.main(["summary", "--stream", "e2e", "--quiet"], deps=st.deps)
    check("the CLI exits 3 (busy) on a held lease", rc == 3, rc)
    lease.unlink()
    rc = ST.main(["verify", "--stream", "e2e"], deps=st.deps)
    check("verify: the chain and every file it names check out", rc == 0)
    ns = ST.build_parser().parse_args(["build", "--stream", "weed_stream_v1", "--k", "4", "--recipes", "r0,x1a"])
    check("build_parser() reads back the L18 argv the autopilot writes (S4)",
          (ns.cmd, ns.stream, ns.k, ns.recipes) == ("build", "weed_stream_v1", 4, "r0,x1a"))
    for argv in (["commit", "--exp", "e2e_s001"], ["milestone", "--stream", "e2e"], ["fork", "--stream", "e2e", "--m", "12"],
                 ["feasibility", "--stream", "e2e", "--holdout", "tsw22", "--m", "6"],
                 ["feasibility", "--stream", "e2e", "--holdout", "tsw22", "--m", "6", "--recipes", "r0,x1a"],
                 ["build", "--stream", "e2e", "--k", "2", "--recipes", "r0,x1a", "--arch", "yolo11s", "--imgsz",
                  "640", "--truth-every", "2"],
                 ["bisect", "--stream", "e2e", "--from", "P_1"], ["rollback", "--stream", "e2e", "--to", "P_1"],
                 ["quarantine", "--source", "x", "--cite", "D28"], ["release", "--stream", "e2e", "--hold", "funnel_F9"]):
        ST.build_parser().parse_args(argv)
    check("the parser takes every argv form of the autopilot's levers (stream_levers.json L18 with --arch, "
          "--imgsz, --truth-every; L28 with --recipes)", True)
    # the dHash cache is a cache: a torn line never refuses the stream, and only the lease holder writes it
    cache = st.p.dhash_cache
    cache.write_bytes((cache.read_bytes() if cache.exists() else b"") + b'{"image": "/torn\n{"image": "/x", "sha256"\n')
    size = cache.stat().st_size
    st2 = ST.Stream("e2e", deps=st.deps, quiet=True)
    st2.load()
    try:
        st2._hash_cache()
        st2.cut_plan(1)
        torn_ok = True
    except ST.StreamError as e:
        torn_ok = str(e)
    st2._cache_add([{"image": "/nowhere.png", "sha256": "0" * 64, "dhash": 1, "variants": None}])
    rc_cut = ST.main(["cut", "--stream", "e2e", "--quiet"], deps=st.deps)
    check("a torn dhash_cache line is skipped, never a refusal of the stream; a reader without the lease (the "
          "dry-run cut) never appends to it", torn_ok is True and cache.stat().st_size == size
          and ("/nowhere.png", "0" * 64) in st2._hash_cache() and rc_cut in (0, 1), (torn_ok, rc_cut))
    check("K beyond K_max is refused", ST.main(["build", "--stream", "e2e", "--k", "9"], deps=st.deps) == 1)
    check("quarantine without --stream refuses when several streams exist; with a cite it is recorded; lifting "
          "needs a person",
          ST.main(["quarantine", "--source", "srcX", "--cite", "D28"], deps=st.deps) == 1
          and ST.main(["quarantine", "--stream", "e2e", "--source", "srcX", "--cite", "no"], deps=st.deps) == 1
          and ST.main(["quarantine", "--stream", "e2e", "--source", "srcX", "--cite", "D28"], deps=st.deps) == 0
          and "srcX" in st.load().q_sources
          and ST.main(["unquarantine", "--stream", "e2e", "--source", "srcX"], deps=st.deps) == 1
          and ST.main(["unquarantine", "--stream", "e2e", "--source", "srcX", "--decided-by", "human:owner"],
                      deps=st.deps) == 0 and "srcX" not in st.load().q_sources)
    saved = ST._inc2
    ST._inc2 = lambda name: None if name == "step1_stream" else saved(name)
    try:
        st.load()
        no_reader = raises(lambda: st.cut_plan(1), ST.StreamError, "step1_stream")
        q_err = (st.summary().get("queue") or {}).get("error") or ""
    finally:
        ST._inc2 = saved
    check("without inc2.step1_stream (the queue's one reader) the queue is not read: the cut refuses and the "
          "summary says why (no private second fold)", no_reader and "step1_stream" in q_err, q_err)
    check("L24 cites only a firing D28 or D31; another diagnosis needs a person",
          ST.main(["quarantine", "--stream", "e2e", "--source", "srcY", "--cite", "D5"], deps=st.deps) == 1
          and "srcY" not in st.load().q_sources
          and ST.main(["quarantine", "--stream", "e2e", "--source", "srcY", "--cite", "D31"], deps=st.deps) == 0
          and ST.main(["quarantine", "--stream", "e2e", "--source", "srcZ", "--cite", "D5", "--decided-by",
                       "human:owner"], deps=st.deps) == 0)
    data = st.p.ledger.read_bytes()
    st.p.ledger.write_bytes(data.replace(b'"event": "cut"', b'"event": "cux"', 1))
    check("a broken ledger chain is refused by every verb", ST.main(["verify", "--stream", "e2e"], deps=st.deps) == 1
          and raises(lambda: st.load(), ST.StreamError, "broken"))
    st.p.ledger.write_bytes(data)


def test_withdraw_fork_feasibility(base, base_rows, guard):
    print("withdraw, feasibility (Stage C), fork")
    M = 6
    ex = Executor(M)
    fb = D.FakeBackend()
    s1 = Step1World("s1_misc")
    mixed = [1, 1, 0, 0, 0, 1] + [0] * 6
    s1.add("good_c", "b0001", mixed, n=M, capture_group="g")
    s1.commit()
    st = new_stream("misc", base, M, s1, guard, fb=fb, stage_b="r0,x1a")
    exp = st.build(1)
    f = st.load()
    keys = list(f.inc_rows(f.segments[1]["increments"][0]))
    d = json.loads(D.Paths(exp).exp_json.read_text())
    check("Stage B: segment 1 runs r0 and the Stage A survivor, truth on", sorted(d["recipes"]) == ["r0", "x1a"]
          and d["truth"] is True)
    check("withdraw needs a reason", raises(lambda: st.withdraw(exp, ""), ST.StreamError))
    st.withdraw(exp, "cancelled by a person (test)", decided_by="human:owner")
    f = st.load()
    check("withdraw: the segment is withdrawn and its images are eligible again, uncounted",
          f.segments[1]["state"] == "withdrawn"
          and all(f.keys[k]["status"] == "released" and f.keys[k]["returns"] == 0 for k in keys))
    check("a withdrawn segment is never committed", raises(lambda: st.commit(exp), ST.StreamError))
    check("Stage C at another M than the stream's is refused, nothing recorded (it measures the stream's own M)",
          raises(lambda: st.feasibility("tsw22", m=M + 1), ST.StreamError, "not this stream's M")
          and ST.main(["feasibility", "--stream", "misc", "--holdout", "tsw22", "--m", str(M + 1), "--recipes",
                       "r0,x1a"], deps=st.deps) == 1 and st.load().feasibility is None)
    cexp = st.feasibility("tsw22", m=M, recipes="r0,x1a")
    cd = json.loads(D.Paths(cexp).exp_json.read_text())
    Drows = C.read_manifest(cd["steps"][0]["manifest"])
    brows = C.read_manifest(cd["base"]["manifest"])
    check("Stage C: D = exactly M tsw22 target images in whole sessions (one split), base = P_0 minus D",
          len(Drows) == M and all(r["key"].startswith("tsw22__") for r in Drows)
          and len(brows) == len(base_rows) - M and not ({r["key"] for r in Drows} & {r["key"] for r in brows})
          and cd["truth"] is True and cd["steps"][0]["clean"] is False and sorted(cd["recipes"]) == ["r0", "x1a"],
          (len(Drows), cd["stream"]))
    p0_ro = st.load().pools["P_0"]["research_only"]
    check("§8: Stage C records its models' research-only flag, P_0's (base and D are drawn from it)",
          (cd["stream"].get("research_only") or {}).get("pool") == p0_ro
          and cd["stream"]["research_only"]["models"] in (True, False, "unknown")
          and cd["stream"]["research_only"]["models"] == (True if p0_ro is True else
                                                          ("unknown" if p0_ro == "unknown" else False)),
          (cd["stream"].get("research_only"), p0_ro))
    drive(cexp, fb, ex)
    res = st.compare(cexp)
    check("Stage C read under Protocol v3: M feasible (the known-good increment is ACCEPTed)",
          res["m_feasible"] is True and not res["species_only_reject"], res)
    check("Stage C runs once per stream version", raises(lambda: st.feasibility("tsw22", m=M), ST.StreamError, "once"))
    s1.add("good_d", "b0002", mixed, n=M, capture_group="nc", research_only=True)
    s1.commit()
    exp2 = st.build(2)
    d2 = json.loads(D.Paths(exp2).exp_json.read_text())
    ro = d2["stream"]["research_only"]
    check("§8: an increment with research_only rows flags the segment's models research_only",
          ro["increments_research_only_rows"] == M and ro["models"] is True, ro)
    drive(exp2, fb, ex)
    res2 = st.commit(exp2)
    f = st.load()
    from weed_optimizer_framework.tools.inc2 import recipes as RC
    gates = [json.loads(ln) for ln in D.Paths(exp2).ledger.read_text().splitlines() if ln.strip()]
    again = RC.stage_b_choice([e for e in gates if e.get("type") == "gate"], {"r0": "r0", "x1a": "x1a"})
    check("Stage B commit with two chains: inc2.recipes.stage_b_choice picks one (recorded), and P_s counts its "
          "research_only rows",
          res2["chosen"] == again["chosen"] and f.segments[2]["commit"]["choice"]["rule"] == RC.STAGE_B_RULE
          and res2["pool"]["research_only_rows"] == M and res2["pool"]["research_only"] is True
          and f.chosen_recipe == res2["chosen"], (res2["chosen"], res2["pool"]))
    check("after Stage B a segment runs the chosen recipe alone", st.recipes_for(3) == [res2["chosen"]]
          and raises(lambda: st.recipes_for(3, ["r0", "x1a"]), ST.StreamError, "pre-registration"))
    tp = st.truth_policy(1, ["r0"], 60000, 6000)
    tp0 = st.truth_policy(0, ["r0"], 60000, 6000)
    tps = st.truth_policy(1, ["r0"], 7626, 763)
    check("L-4 truth policy: a step above 25 GPU-h runs the truth arm on every ceil(cost/25)-th segment "
          "(the first always); at base_v2's size every segment has it",
          tp["every"] >= 2 and tp["on"] is False and tp0["on"] is True and tps["every"] == 1 and tps["on"] is True,
          (tp, tps))
    check("switching the truth arm off is a person's decision (P3)",
          raises(lambda: st.truth_policy(0, ["r0"], 100, 10, no_truth=True), ST.StreamError, "person")
          and st.truth_policy(0, ["r0"], 100, 10, no_truth=True, decided_by="human:owner")["on"] is False)
    tpr = st.truth_policy(1, ["r0"], 60000, 6000, requested_every=1)
    tpl = st.truth_policy(1, ["r0"], 60000, 6000, requested_every=tp["every"] + 5)
    check("the autopilot's --truth-every can only make the truth arm more frequent (both recorded)",
          tpr["on"] is True and tpr["every"] == 1 and tpr["every_stream"] == tp["every"]
          and tpl["every"] == tp["every"] and tpl["on"] is tp["on"], (tpr, tpl))
    check("the arm is fixed once a segment is built", raises(lambda: st.choose_arm(TMP / "none.json"), ST.StreamError,
                                                             "fixed"))
    # a person's rollback to a pool without a milestone: the bisection could never decide, so it is not built
    st.rollback("P_0", decided_by="human:owner")
    n_led = len(st.ledger.read())
    check("bisect (L27) refuses before building when P_c has no good milestone to compare with (X4 stays)",
          raises(lambda: st.bisect("P_0"), ST.StreamError, "no good milestone") and len(st.ledger.read()) == n_led
          and not D.Paths(ST.bisect_exp("misc", 1)).exp_json.exists())
    # a person's releases survive the fork
    f9 = s1.add("good_f", "b0003", mixed, n=M, capture_group="f9", holds=("funnel_F9",))
    s1.commit()
    st.release("funnel_F9", "owner: release before F9", "human:owner")
    check("a fork that is not a doubling needs a person (X17)", raises(lambda: st.fork(M + 1), ST.StreamError, "X17"))
    new = st.fork(2 * M)
    f = st.load()
    nst = ST.Stream(new, deps=st.deps, quiet=True)
    nf = nst.load()
    check("L22 fork: a new stream version with M doubled on the current pool's bytes; the old one builds nothing; "
          "its name cannot be read as a stream experiment",
          nf.defn["M"] == 2 * M and nf.current_pool()["sha256"] == f.current_pool()["sha256"]
          and f.forked_to == new and raises(lambda: st.build(1), ST.StreamError, "forked")
          and new == "misc_fork%d" % (2 * M) and raises(lambda: ST.sid_of_exp(new), ST.StreamError), (new, nf.defn["M"]))
    _p, _r, anaf = nst.cut_plan(1)
    check("the fork keeps a person's releases: the funnel_F9 rows released in the old stream are eligible in the "
          "new one", {r["key"] for r in f9} <= {e["key"] for e in anaf["eligible"]}, anaf["reasons"])


def test_crash_repair(base, guard):
    print("a killed operation is completed by the next writer (its ledger line comes before the external step)")
    M = 6
    ex = Executor(M)
    fb = D.FakeBackend()
    clock = Clock()
    mixed = [1, 1, 0, 0, 0, 1] + [0] * 6
    s1 = Step1World("s1_crash")
    s1.add("good_k", "b0001", mixed, n=M, capture_group="k")
    s1.commit()
    bdl = BaselineDouble(fb)
    st = new_stream("crashw", base, M, s1, guard, fb=fb, clock=clock, baseline=bdl)
    real_event = st.event

    def killed_at(name):
        def ev(event, by="platform", **kw):
            if event == name:
                raise KeyboardInterrupt("killed before the %s line" % name)
            return real_event(event, by=by, **kw)
        return ev
    # (a) killed after the cut lines, before the build line
    st.event = killed_at("build")
    try:
        st.build(1)
    except KeyboardInterrupt:
        pass
    st.event = real_event
    f = st.load()
    stranded = sorted(k for k, v in f.keys.items() if v["status"] == "in_increment")
    ok_a0 = len(stranded) == M and not f.segments
    rc = ST.main(["summary", "--stream", "crashw", "--quiet"], deps=st.deps, clock=clock)
    f = st.load()
    _p, _r, ana = st.cut_plan(1)
    check("cut lines without their build line: the next writer withdraws the increment and releases its images "
          "(uncounted), which are cut again", ok_a0 and rc == 0 and all(f.keys[k]["status"] == "released"
                                                                        for k in stranded)
          and sorted(r["key"] for r in _p[0]["rows"]) == stranded, (ok_a0, rc, _r))
    # (b) killed inside driver init (no state.json, nothing submitted)
    real_init = D.Driver.init

    def boom(self, defn):
        raise KeyboardInterrupt("killed in driver init")
    D.Driver.init = boom
    try:
        st.build(1)
    except KeyboardInterrupt:
        pass
    finally:
        D.Driver.init = real_init
    exp_b = st.load().in_flight()[0]["exp"]
    ST.main(["summary", "--stream", "crashw", "--quiet"], deps=st.deps, clock=clock)
    within = next((s["state"] for s in st.load().segments.values() if s["exp"] == exp_b), None)
    clock.t += ST.CRASH_GRACE_S + 1
    ST.main(["summary", "--stream", "crashw", "--quiet"], deps=st.deps, clock=clock)
    f = st.load()
    seg_b = next(s for s in f.segments.values() if s["exp"] == exp_b)
    check("a built segment whose driver init never wrote state.json stays in flight within the grace, then is "
          "withdrawn and its images released; the next build proceeds",
          within == "built" and seg_b["state"] == "withdrawn"
          and all(f.keys[k]["status"] == "released" for k in stranded) and st.build(1) != exp_b, (within, seg_b))
    # (c) a milestone whose inc2.baseline build was killed
    def killed_runner(argv):
        raise KeyboardInterrupt("killed in inc2.baseline build")
    st.deps.baseline_runner = killed_runner
    try:
        st.milestone()
    except KeyboardInterrupt:
        pass
    st.deps.baseline_runner = bdl
    waiting = st.milestone()
    clock.t += ST.CRASH_GRACE_S + 1
    built = st.milestone()
    f = st.load()
    check("a milestone whose build never completed is recorded build_failed after the grace, and L20 builds the "
          "next one (it never waits on it for ever)",
          waiting.get("waiting") == ST.milestone_exp("crashw", 1) and f.milestones[1]["state"] == "failed"
          and built == {"built": ST.milestone_exp("crashw", 2)}, (waiting, built))
    # (d) Stage C killed in driver init: built again after the grace
    D.Driver.init = boom
    try:
        st.feasibility("tsw22", m=M)
    except KeyboardInterrupt:
        pass
    finally:
        D.Driver.init = real_init
    once = raises(lambda: st.feasibility("tsw22", m=M), ST.StreamError, "once")
    clock.t += ST.CRASH_GRACE_S + 1
    ST.main(["summary", "--stream", "crashw", "--quiet"], deps=st.deps, clock=clock)
    again = st.feasibility("tsw22", m=M)
    check("Stage C whose driver init never completed is refused as a rerun within the grace, then built again",
          once and again == ST.feasibility_exp("crashw") and D.Paths(again).state.exists())
    del ex


def test_lock_pin(base, guard):
    print("the cut clears rows against the LOCK v2 the stream was defined on")
    d = TMP / "lockw_splits"
    d.mkdir(parents=True, exist_ok=True)
    lock = d / "LOCK.json"
    lock.write_text(json.dumps({"splits_version": "v2", "manifests": {"base_v2": C.sha256_file(base)}}))
    s1 = Step1World("s1_lock")
    s1.add("srcL", "b0001", [1, 1, 0, 0, 0, 1] + [0] * 6, n=3, capture_group="l")
    s1.commit()
    st = ST.Stream("lockw", deps=ST.Deps(guard=guard, hasher=hashes, backend=D.FakeBackend()), clock=Clock(),
                   quiet=True)
    st.init(base=base, lock=lock, m=3, testing=True, step1_dir=str(s1.root))
    st.load()
    plans, _r, _a = st.cut_plan(1)
    lock.write_text(json.dumps({"splits_version": "v2", "manifests": {"base_v2": C.sha256_file(base)},
                                "relocked": True}))
    st.load()
    check("a LOCK v2 changed since init refuses the cut (another splits version is another stream version)",
          len(plans) == 1 and raises(lambda: st.cut_plan(1), ST.StreamError, "LOCK v2"))


# ----------------------------------------------------------- job script
SCRIPT = PKG_ROOT / "run_inc2_build.sh"


def test_build_script():
    print("run_inc2_build.sh")
    text = SCRIPT.read_text()
    check("bash -n accepts it", subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0)
    check("GPU-shared with one V100 (RM-shared is refused by the allocation)",
          "#SBATCH --partition=GPU-shared" in text and "gpu:v100-32:1" in text)
    check("no git reset, no copy of the nested package over the outer one, no Roboflow sync",
          "git reset" not in text and "rsync" not in text and "roboflow" not in text.lower()
          and not any(ln.strip().startswith("cp ") for ln in text.splitlines()))
    check("the executor is v2's: INC_JOB_SCRIPT is exported as run_inc2_job.sh", "run_inc2_job.sh" in text
          and "export INC_JOB_SCRIPT" in text)
    for m in ("tools/inc2/stream.py", "tools/inc2/stream_report.py", "tools/inc2/baseline.py", "tools/inc2/splits.py",
              "tools/inc2/pilot4.py", "tools/inc/driver.py", "tools/inc/gate.py", "tools/inc/scorer.py"):
        check("the drift check hashes %s" % m, m in text)
    import re
    listed = set(re.findall(r"tools/[A-Za-z0-9_/]+\.(?:py|json)", text))
    # what the builders import (inc2.splits' copy scan loads funnel.embed / leak / estimate lazily; funnel.embed
    # loads semisup_labeler lazily), computed from the real modules, not restated
    code = ("import importlib, sys\n"
            "for m in ('splits', 'baseline', 'pilot4', 'stream', 'stream_report', 'step1_stream', 'guard', 'gate3',"
            " 'recipes', 'base3', 'scorer_agnostic', 'mask', 'eval_hits', 'train'):\n"
            "    importlib.import_module('weed_optimizer_framework.tools.inc2.' + m)\n"
            "for m in ('funnel.embed', 'funnel.leak', 'funnel.estimate', 'semisup_labeler'):\n"
            "    importlib.import_module('weed_optimizer_framework.tools.' + m)\n"
            "print(' '.join(sorted(k for k in sys.modules if k.startswith('weed_optimizer_framework.tools.'))))\n")
    pr = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(PKG_ROOT), timeout=300,
                        env=dict(os.environ, PYTHONPATH=str(PKG_ROOT)))
    need = set()
    for mod in pr.stdout.split():
        rel = mod[len("weed_optimizer_framework.tools."):].split(".")
        if rel[0] not in ("inc", "inc2", "funnel", "cwd12_species", "near_dup", "mega_trainer", "semisup_labeler"):
            continue
        pkg_dir = PKG_ROOT / "weed_optimizer_framework" / "tools" / "/".join(rel)
        need.add("tools/%s/__init__.py" % "/".join(rel) if pkg_dir.is_dir() else "tools/%s.py" % "/".join(rel))
    need.add("tools/funnel/domains/weed.json")        # the domain config the splits build and the scan read
    from weed_optimizer_framework.tools.inc2 import base3 as B3
    conf_rel = "tools/inc2/%s" % B3.CONFIG.name                  # the config inc2.base3 loads (base3_v2.json)
    need.update((conf_rel, "tools/inc_autopilot/stream_thresholds.json"))   # inc2.base3 reads both
    check("the drift check hashes every module the builders import (including the copy scan's funnel.embed, leak, "
          "estimate, domain, ledger, semisup_labeler) and the funnel domain config",
          pr.returncode == 0 and need and not (need - listed), (pr.stderr[-300:], sorted(need - listed)))
    check("the drift check hashes the base v3 config the builder loads (%s), not the one it no longer reads "
          "(base3_v1.json)" % conf_rel, conf_rel == "tools/inc2/base3_v2.json" and conf_rel in listed
          and "tools/inc2/base3_v1.json" not in listed, sorted(x for x in listed if "base3" in x))
    hdr = dict(re.findall(r"^#SBATCH --([a-z-]+)=(\S+)", text, re.M))
    check("the time limit is 12 h, for every verb (sbatch reads it at submission; the base v3 build needs more than "
          "the old 4 h), and the script states why and what it costs the other verbs",
          hdr.get("time") == "12:00:00" and "Time limit: 12 h for every verb" in text and "backfill" in text, hdr)
    repo = TMP / "sh_repo"
    inc = TMP / "sh_inc"
    shims = TMP / "sh_shims"
    shims.mkdir(parents=True, exist_ok=True)
    (shims / "python").write_text("#!/bin/bash\nexec %s \"$@\"\n" % sys.executable)
    (shims / "squeue").write_text("#!/bin/bash\nexit 0\n")
    if not shutil.which("sha256sum"):
        (shims / "sha256sum").write_text("#!/bin/bash\nexec %s -c 'import hashlib,sys; p=sys.argv[1]; "
                                         "print(hashlib.sha256(open(p,\"rb\").read()).hexdigest(), p)' \"$@\"\n"
                                         % sys.executable)
    for p in shims.iterdir():
        p.chmod(0o755)
    conda = TMP / "sh_conda.sh"
    conda.write_text("conda() { return 0; }\n")
    stub_mod = ("import json, os, sys\na = sys.argv[1:]\nprint('[stub %(m)s] ' + ' '.join(a), flush=True)\n"
                "print('INC_JOB_SCRIPT=' + os.environ.get('INC_JOB_SCRIPT', ''), flush=True)\n"
                "print('HF_HUB_OFFLINE=' + os.environ.get('HF_HUB_OFFLINE', ''), flush=True)\n"
                "if 'refuse' in ' '.join(a):\n    print('[inc2.%(m)s] ERROR: refused for the test', file=sys.stderr)\n"
                "    sys.exit(1)\n"
                "if '%(m)s' == 'stream' and a[0] == 'build':\n    print('[inc2.stream] built experiment wsv_s001')\n")
    stub_driver = "import sys\nprint('[stub driver] ' + ' '.join(sys.argv[1:]))\n"
    for root in (repo / "weed_optimizer_framework", repo / "weed_llm_benchmark" / "weed_optimizer_framework"):
        for d in ("", "tools", "tools/inc", "tools/inc2"):
            (root / d).mkdir(parents=True, exist_ok=True)
            (root / d / "__init__.py").write_text("")
        for m in listed:
            p = root / m
            p.parent.mkdir(parents=True, exist_ok=True)
            base = os.path.basename(m)[:-3]
            if m.startswith("tools/inc2/") and base in ("stream", "splits", "baseline", "pilot4", "base3"):
                p.write_text(stub_mod % {"m": base})
            elif m == "tools/inc/driver.py":
                p.write_text(stub_driver)
            else:
                p.write_text("# stub %s\n" % m)

    def run(args, extra=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("INCAP_", "INC_", "SLURM_"))}
        env.update({"PATH": "%s:%s" % (shims, os.environ.get("PATH", "")), "INC_BUILD_REPO": str(repo),
                    "INC_BUILD_CONDA_SH": str(conda), "INC_DIR": str(inc), "SLURM_JOB_ID": "901",
                    "TMPDIR": str(TMP)})
        env.update(extra or {})
        return subprocess.run(["bash", str(SCRIPT)] + args, capture_output=True, text=True, env=env, timeout=120)

    def prov(name):
        p = inc / "_campaign" / "provenance" / ("%s.json" % name)
        return json.loads(p.read_text()) if p.exists() else None

    p = run(["inc2.stream", "build", "--stream", "wsv", "--k", "2"])
    a = ((prov("stream_wsv") or {}).get("attempts") or [{}])[-1]
    check("inc2.stream build: runs the module with INC_JOB_SCRIPT = run_inc2_job.sh, then advances the experiment "
          "it built (parsed from its output)",
          p.returncode == 0 and "[stub stream] build --stream wsv --k 2" in p.stdout
          and "INC_JOB_SCRIPT=%s" % (repo / "weed_llm_benchmark" / "run_inc2_job.sh") in p.stdout
          and "[stub driver] advance --exp wsv_s001" in p.stdout and a.get("status") == "advanced"
          and a.get("built_exp") == "wsv_s001", (p.returncode, p.stdout[-600:], p.stderr[-600:], a))
    p = run(["inc2.baseline", "build", "--exp", "b_v2", "--manifest", "/x/base_v2.jsonl", "--seeds", "0,1,2,3,4"])
    check("inc2.baseline build: the lock and provenance are named by --exp; the advance follows",
          p.returncode == 0 and "[stub driver] advance --exp b_v2" in p.stdout
          and (prov("b_v2") or {}).get("attempts", [{}])[-1].get("status") == "advanced", p.stdout[-400:])
    p = run(["inc2.baseline", "rescore-native", "--exp", "b_v2_m832", "--reference", "b_v2_m640"])
    a = ((prov("native_b_v2_m832") or {}).get("attempts") or [{}])[-1]
    check("inc2.baseline rescore-native (L23N, 2026-10-01): run, recorded as scored under native_<exp>, never the "
          "arm's own provenance, and no advance (it builds nothing)",
          p.returncode == 0 and "[stub baseline] rescore-native --exp b_v2_m832 --reference b_v2_m640" in p.stdout
          and "[stub driver]" not in p.stdout and a.get("status") == "scored" and a.get("build_rc") == 0
          and len((prov("b_v2") or {}).get("attempts") or []) == 1 and prov("b_v2_m832") is None,
          (p.returncode, p.stdout[-400:], a))
    p = run(["inc2.baseline", "rescore-native", "--exp", "refuse", "--reference", "b_v2_m640"])
    a = ((prov("native_refuse") or {}).get("attempts") or [{}])[-1]
    check("  a rescore-native refusal: exit 1, recorded build_failed with its ERROR line, no advance",
          p.returncode == 1 and a.get("status") == "build_failed" and "ERROR" in str(a.get("refusal"))
          and "[stub driver]" not in p.stdout, (p.returncode, a))
    p = run(["inc2.baseline", "rescore-agnostic", "--exp", "e1_b_m640", "--reference", "e1_a_m640"])
    a = ((prov("agnostic_e1_b_m640") or {}).get("attempts") or [{}])[-1]
    check("inc2.baseline rescore-agnostic (L23E, 2026-10-03): run, recorded as scored under agnostic_<exp>, no "
          "advance (it builds nothing)",
          p.returncode == 0 and "[stub baseline] rescore-agnostic --exp e1_b_m640 --reference e1_a_m640" in p.stdout
          and "[stub driver]" not in p.stdout and a.get("status") == "scored" and prov("e1_b_m640") is None,
          (p.returncode, p.stdout[-400:], a))
    p = run(["inc2.baseline", "rescore-e2"])
    a = ((prov("e2_v1") or {}).get("attempts") or [{}])[-1]
    check("inc2.baseline rescore-e2 (L23C, 2026-10-04): run without --exp, recorded as scored under e2_v1, no advance "
          "(it builds nothing)",
          p.returncode == 0 and "[stub baseline] rescore-e2" in p.stdout and "[stub driver]" not in p.stdout
          and a.get("status") == "scored" and a.get("build_rc") == 0, (p.returncode, p.stdout[-400:], a))
    p = run(["inc2.base3", "build", "--stream", "wsv"])
    a = ((prov("base3_v3") or {}).get("attempts") or [{}])[-1]
    check("inc2.base3 build (L23V, 2026-10-03): run under the provenance and lock base3_v3 with HF_HUB_OFFLINE=1, "
          "recorded built_splits, no advance (it builds no experiment)",
          p.returncode == 0 and "[stub base3] build --stream wsv" in p.stdout and "[stub driver]" not in p.stdout
          and a.get("status") == "built_splits" and "HF_HUB_OFFLINE=1" in p.stdout, (p.returncode, p.stdout[-500:],
                                                                                    p.stderr[-300:], a))
    p = run(["inc2.base3", "build"])
    check("  inc2.base3 build without --stream: usage error, nothing run", p.returncode == 2 and "[stub" not in p.stdout,
          (p.returncode, p.stderr[-300:]))
    p = run(["inc2.stream", "commit", "--exp", "wsv_s001"])
    check("inc2.stream commit is not a build verb: usage error, nothing run",
          p.returncode == 2 and "usage:" in p.stderr and "[stub" not in p.stdout, (p.returncode, p.stderr[-300:]))
    p = run(["inc2.stream", "milestone", "--stream", "wsv"])
    check("inc2.stream milestone is a build verb", p.returncode == 0 and "[stub stream] milestone" in p.stdout)
    p = run(["inc2.splits", "build", "--exp", "refuse"])
    a = ((prov("refuse") or {}).get("attempts") or [{}])[-1]
    check("a builder refusal: its exit status and ERROR line recorded, no advance",
          p.returncode == 1 and a.get("status") == "build_failed" and "[stub driver]" not in p.stdout,
          (p.returncode, a))
    (repo / "weed_optimizer_framework" / "tools" / "inc2" / "stream.py").write_text("# outer copy edited\n")
    p = run(["inc2.stream", "build", "--stream", "wsv", "--k", "1"])
    check("outer / nested drift of an inc2 module: refused before the builder",
          p.returncode == 1 and "FATAL: outer modules differ" in p.stderr and "[stub" not in p.stdout,
          (p.returncode, p.stderr[-300:]))
    for args in (["inc.pilot", "build", "--exp", "x"], ["inc2.stream"], [], ["inc2.stream", "build", "--stream", "../x"]):
        p = run(args)
        check("usage error for %s: exit 2" % (args or "nothing"), p.returncode == 2 and "usage:" in p.stderr,
              (p.returncode, p.stderr[-200:]))


def main():
    base, base_rows = make_base()
    dev_paths = [WORLD / "dev" / ("dev_%d.png" % i) for i in range(3)]
    for i, p in enumerate(dev_paths):
        save_pattern(pattern(4242 + i), p)
    guard = make_guard(dev_paths, base_rows)
    test_run_as_main()
    test_rules()
    test_ledger()
    test_init(base, base_rows, guard)
    test_cut(base, base_rows, guard, dev_paths)
    test_licence_release(base, guard)
    test_guard_at_cut(base, base_rows, guard, dev_paths)
    test_test_v1_never_cut(base, guard)
    test_deadlock(base, guard)
    test_split_units(base, guard)
    test_choose_arm(base, guard)
    st, _fb, _ex = test_end_to_end(base, base_rows, guard)
    test_test_blindness(st)
    test_lease_and_cli(st)
    test_withdraw_fork_feasibility(base, base_rows, guard)
    test_crash_repair(base, guard)
    test_lock_pin(base, guard)
    test_build_script()
    print("\n%d failure(s)" % len(FAILURES))
    if not FAILURES:
        shutil.rmtree(TMP, ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
