#!/usr/bin/env python3
"""E2: the 12-class detector on E1-B's weed-box backbone (inc2/recipes.py
E2_*, e2_recipe, e2_cost; inc2/train.py e2_problems, init_check on the
record, the environment check and init_transfer; inc2.baseline build --e2,
rescore-e2, e2-verdict, e2-test-read, e2-test-report; docs/CONTINUOUS_LOOP.md,
"Amendment (2026-10-04): E2, the 12-class detector on E1-B's backbone
(pre-registered)").

The synthetic world of tests/test_inc2_train.py (v1 exams materialised, the
v1 LOCK with scorer.py's sha256, LOCK v2 marked testing). E1-B is a fixture
(e1_b_m640: E1's arm B on m640 with cold_budget, three done base runs whose
weights are real 13-class checkpoints, production run records under the
world's LOCK v2), E1's verdict a decided, qualifying capacity/e1_v1.json, and
the reference b_v2_m640 is built by the real builder (role capacity, m640)
with base runs that record this machine's Ultralytics and torch. Training and
scoring are real Ultralytics passes on the CPU in test mode.

What is pinned:
- the constants (arms W cold and S x1b, the six experiment names, seeds 0-2,
  the seed text inc2/e2/species_se, 1,000 resamples, E1's format equal to
  inc2.baseline's), x1b differing from cold in epochs and warmup_bias_lr
  only, e2_cost's image-epochs at 14.2 ms;
- recipes: x1b accepted for a base run only when named, refused for a union
  or an incremental run and under another name; None, cold and cold_budget
  as before;
- the build: e2_w_m640_seed0 starts from E1-B's base__s0 final.pt (absolute
  path, its run.json's sha256), its e2 record names the reference and what
  differs from it, finals dev and imageweeds, E2-S names x1b, research-only
  inherited from E1-B (fail closed), priced from image-epochs, and the
  pinned driver writes the base run's spec with that init and seed 0; each
  refusal (the arm letter, a union, the role, the arm, the seeds, the name,
  the manifest, the finals, the reference missing or differing, E1's verdict
  missing, pending, not qualifying or naming another experiment, E1-B not
  arm B, its run not done, its final.pt missing, a symlink or modified)
  leaves no directory of the experiment;
- inc2.train in production (stops at device, no CUDA here): both arms pass
  stage recipe from the recorded init; the wrong recipe for either arm, x1b
  without an e2 record, the arm's checkpoint as init, an init changed after
  the build, another seed, E1-B's run.json changed, E1's verdict no longer
  qualifying, another training environment and a union run are refused at
  stage recipe; E1's verdict rewritten with only generated_utc changed is
  not; init_weights other than the arm's checkpoint needs an e2 record that
  names it; a final run under an E2 exp.json passes stage recipe;
- the whole load: a 1-epoch CPU run from E1-B's weights records every tensor
  equal; an init with a 12-class head records the differing head tensors
  (testing); init_transfer compares the state_dict (a BatchNorm buffer
  counts), unwraps a DDP-style wrapper, and require_whole_load refuses in
  production;
- the bootstrap with seed text inc2/e2/species_se equals an independent
  recomputation and is deterministic;
- the verdict rule on synthetic native files: each condition alone fails,
  pooled sd, the larger D chosen, a tie to S, a missing file pending,
  every refusal (a stamp, a test-mode file, an unchecked protocol score,
  weights or init that do not match, two experiments of one seed, another
  reference manifest), the decision file dev only;
- e2_verdict keeps a decided file byte for byte when its decision agrees
  (also when a reported-only protocol score changed), refuses one that
  disagrees and overwrites a pending one;
- rescore-e2 end to end with real CPU passes (written once, a complete
  record of names and sha256s, a second run keeps everything, an undone
  final run refuses before anything is written);
- e2-test-read once per qualifying arm, after the verdict (one decided under
  the pre-registered parameters), checking everything before writing;
  e2-test-report pending, then the means and the gap to 0.90 (the chosen
  arm the headline), refusing a test-mode score and any score that is not
  the prepared read's (its weights, the read record, the shared scorer,
  test manifest and key order); neither file on the evidence;
- the CLI's refusals and a build through it.
rescore_native after the _native_one refactor: tests/test_inc2_native.py.

Run:  python3 tests/test_inc2_e2.py
"""
import contextlib
import copy
import json
import math
import os
import pathlib
import shutil
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_train as W  # noqa: E402  (sets INC_DIR / REPO / INC_SCORER_TESTING first)
import test_inc2_native as N  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import driver as D  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc2 import baseline as B  # noqa: E402
from weed_optimizer_framework.tools.inc2 import recipes as RC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import scorer_native as SN  # noqa: E402
from weed_optimizer_framework.tools.inc2 import scorer_sidecar as SC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import train as T  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import brain_plan as BP  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import evidence as E  # noqa: E402

FAILURES = W.FAILURES
check = W.check
CPU32 = {"batch": 32, "device": "cpu"}
E1B = "e1_b_m640"
REF = RC.E2_REFERENCE_EXP


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except (B.BaselineError, RC.RecipeError, D.DriverError, T.RunError, SN.NativeRefused) as e:
        return e
    return None


def local_env():
    import torch
    import ultralytics
    return {"ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__}


def base_v2():
    return W.v2_dir() / "base_v2.jsonl"


def lock_shas():
    return W.sha(W.v2_dir() / "LOCK.json"), W.sha(W.v2_dir() / "nevertrain_dhash.json")


@contextlib.contextmanager
def saved(*paths):
    """Restore each file (or its absence) afterwards."""
    keep = {}
    for p in paths:
        p = pathlib.Path(p)
        keep[p] = (p.read_bytes(), p.is_symlink(), os.readlink(str(p)) if p.is_symlink() else None) \
            if (p.exists() or p.is_symlink()) else None
    try:
        yield
    finally:
        for p, v in keep.items():
            if p.is_symlink() or p.exists():
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p)
                else:
                    p.unlink()
            if v is not None:
                p.parent.mkdir(parents=True, exist_ok=True)
                if v[1]:
                    os.symlink(v[2], str(p))
                else:
                    p.write_bytes(v[0])


def named_checkpoint(path, seed=0, nc=None):
    """A 13-class yolo11n checkpoint with the INC class names (the scorer accepts it untrained); nc 12 gives a
    checkpoint whose class head Ultralytics cannot load into a 13-class model."""
    import torch
    from ultralytics.nn.tasks import DetectionModel
    if nc is None:
        return N.named_checkpoint(path, seed=seed)
    torch.manual_seed(seed)
    net = DetectionModel("yolo11n.yaml", nc=nc, verbose=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net, "train_args": {}}, path)
    return path


# ------------------------------------------------------------------ fixtures
E1_DECISION = {"format": "inc2-e1-verdict/1", "status": "decided", "exp": E1B, "reference": "e1_a_m640",
               "seeds": [0, 1, 2], "diff": 0.0217, "pooled_sd": 0.0025, "two_pooled_sd": 0.005, "se_diff": 0.0068,
               "qualifies": True, "testing_allowed": False, "summary_sha256": "5" * 64}


def write_e1_verdict(**over):
    p = T.e1_verdict_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = dict(E1_DECISION, generated_utc="2026-10-04T00:00:00Z", **over)
    p.write_text(json.dumps(doc))
    return p


def e1b_run(seed, path=None, nc=None):
    """E1-B's base__s<seed>: its final.pt (a real checkpoint) and a done production run.json under the world's
    LOCK v2 with a clean guard."""
    rd = C.INC_DIR / E1B / "runs" / ("base__s%d" % seed)
    w = rd / "weights" / "final.pt"
    if w.exists() or w.is_symlink():
        w.unlink()
    named_checkpoint(w, seed=50 + seed, nc=nc)
    lock, idx = lock_shas()
    (rd / "run.json").write_text(json.dumps({
        "status": "done", "testing": False, "weights_sha256": W.sha(w), "weights_epoch": 39,
        "guard": {"lock_sha256": lock, "index_sha256": idx, "checked": 300, "refused": 0, "crosscheck_hits": 0}}))
    return w


def make_e1b(flag=True):
    root = C.INC_DIR / E1B
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    defn = {"exp": E1B, "type": "baseline", "role": "baseline", "seeds": [0, 1, 2], "testing": False,
            "final_exams": ["dev", "imageweeds"], "recipe_name": RC.BUDGET_NAME, "base": {"n_images": 300},
            "e1": {"arm": "B", "summary_sha256": "5" * 64}, "research_only": {"flag": flag, "basis": "rows of "
                                                                              "unknown licence"}}
    defn.update(RC.stamp(RC.resolve_arm("m640", require_weights=False)))
    (root / "exp.json").write_text(json.dumps(defn))
    for s in (0, 1, 2):
        e1b_run(s)


def make_reference():
    """b_v2_m640 by the real builder (role capacity, m640, testing): exp.json as the driver writes it, and done
    base runs recording this machine's Ultralytics and torch."""
    root = C.INC_DIR / REF
    if root.exists():
        shutil.rmtree(root)
    defn, _summ = B.build_definition(REF, manifest=base_v2(), role="capacity", arm="m640", testing=CPU32)
    (root / "exp.json").write_text(json.dumps(defn))
    for s in (0, 1, 2):
        rd = root / "runs" / ("base__s%d" % s)
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "run.json").write_text(json.dumps(dict({"status": "done", "testing": False}, **local_env())))
    return defn


def setup_world():
    make_e1b()
    write_e1_verdict()
    make_reference()


def e2_build(letter, seed, testing=None, exp=None, **kw):
    return B.build(exp or RC.e2_exp(letter, seed), manifest=kw.pop("manifest", base_v2()), seeds=kw.pop("seeds", str(seed)),
                   arm=kw.pop("arm", "m640"), role=kw.pop("role", "baseline"), e2=letter,
                   testing=testing if testing is not None else W.TESTING, backend=D.FakeBackend(), quiet=True, **kw)


def drop(exp):
    shutil.rmtree(str(C.INC_DIR / exp), ignore_errors=True)


# ------------------------------------------------------------------ 1, 2
def test_constants_and_recipes():
    print("E2's constants and recipes")
    check("pre-registered: arms W (cold) and S (x1b), the six experiments, seeds 0-2, reference b_v2_m640, the seed "
          "text inc2/e2/species_se, 1,000 resamples, E1's format = inc2.baseline's",
          RC.E2_ARMS == {"W": "cold", "S": "x1b"} and RC.E2_SEEDS == (0, 1, 2) and RC.E2_ARM == "m640"
          and [RC.e2_exp(k, s) for k in ("W", "S") for s in (0, 1, 2)]
          == ["e2_w_m640_seed0", "e2_w_m640_seed1", "e2_w_m640_seed2", "e2_s_m640_seed0", "e2_s_m640_seed1",
              "e2_s_m640_seed2"]
          and RC.E2_REFERENCE_EXP == "b_v2_m640" and B.E2_SEED_TEXT == "inc2/e2/species_se"
          and B.E2_RESAMPLES == 1000 and RC.E2_E1_FORMAT == B.E1_FORMAT and B.e2_exps()["S"][2] == "e2_s_m640_seed2")
    w, s = RC.e2_recipe("W"), RC.e2_recipe("S")
    check("e2_recipe: W is cold('m640'), S is incremental('x1b', 'm640'); they differ in epochs and warmup_bias_lr "
          "only (100 vs 50, 0.1 vs 0.01)",
          w == RC.cold("m640") and s == RC.incremental("x1b", "m640")
          and sorted(k for k in set(w) | set(s) if w.get(k) != s.get(k)) == ["epochs", "warmup_bias_lr"]
          and (w["epochs"], s["epochs"], w["warmup_bias_lr"], s["warmup_bias_lr"]) == (100, 50, 0.1, 0.01))
    check("e2_recipe and e2_exp refuse another arm, letter or seed",
          refused(RC.e2_recipe, "S", "n640") is not None and refused(RC.e2_recipe, "X") is not None
          and refused(RC.e2_exp, "W", 3) is not None and refused(RC.e2_exp, "W", True) is not None)
    cw = RC.e2_cost(6811, "W", [0], ["dev", "imageweeds"])
    cs = RC.e2_cost(6811, "S", [0], ["dev", "imageweeds"])
    check("e2_cost: W 681,100 image-epochs (100 x N), S 340,550 (50 x N), at 14.2 ms (2.69 and 1.34 GPU-h of training)",
          cw["image_epochs"] == 681100 and cs["image_epochs"] == 340550 and cw["ms_per_image_epoch"] == 14.2
          and abs(cw["per_run_gpu_h"][0] - 681100 * 14.2 / 3.6e6 - (cw["per_run_gpu_h"][0] - 2.68656)) < 1e-3
          and abs(681100 * 14.2 / 3.6e6 - 2.6866) < 1e-3 and cw["recipe_name"] == "cold" and cs["recipe_name"] == "x1b"
          and cw["walltime"]["over_d26_line"] == [False, False], (cw, cs))
    print("recipes: x1b on a base run only when named")
    check("deviations('base', x1b, recipe_name='x1b') is [] and match names it x1b",
          RC.deviations("base", s, "m640", recipe_name="x1b") == [] and RC.match("base", s, "m640", "x1b") == "x1b")
    check("named for a union or an incremental run it departs; an unknown name departs; unnamed, x1b on a base run "
          "departs from cold",
          RC.deviations("union", s, "m640", recipe_name="x1b") != []
          and RC.deviations("cand", s, "m640", recipe_name="x1b") != []
          and RC.deviations("base", s, "m640", recipe_name="x9") != []
          and RC.match("base", s, "m640", "x9") is None and RC.deviations("base", s, "m640") != []
          and RC.match("cand", s, "m640", "x1b") is None)
    a = RC.cold_budget("m640", 6811)
    check("None, cold and cold_budget behave as before (E1's cases)",
          RC.deviations("base", a, "m640", "cold_budget", 6811) == [] and len(RC.deviations("base", a, "m640")) == 3
          and RC.deviations("union", a, "m640", "cold_budget", 6811) != []
          and RC.match("base", a, "m640", "cold_budget", 6811) == "cold_budget"
          and RC.match("base", RC.cold("m640"), "m640") == "cold"
          and RC.match("base", RC.cold("m640"), "m640", "cold") == "cold"
          and RC.match("cand", s, "m640") == "x1b" and RC.deviations("cand", s, "m640") == [])


# ------------------------------------------------------------------ 3, 4
def test_build():
    print("inc2.baseline build --e2: refusals first, each leaving nothing")
    e2w0 = RC.e2_exp("W", 0)
    root = C.INC_DIR / e2w0
    cases = []

    def refuse(what, fn=None, frag=None, exp=e2w0, **kw):
        e = refused(fn or e2_build, kw.pop("letter", "W"), kw.pop("seed", 0), exp=exp, **kw)
        ok = e is not None and (frag is None or frag in str(e)) and not (C.INC_DIR / exp).exists()
        cases.append((what, ok, str(e)[:240]))
        drop(exp)
    refuse("the arm letter X", letter="X", frag="E2's arms are W")
    e = refused(B.build_definition, e2w0, union=[W.v2_dir() / "train_core.jsonl", W.v2_dir() / "tsw22.jsonl"],
                role="union", e2="W", testing=W.TESTING)
    cases.append(("a union", e is not None and "never a union" in str(e) and not root.exists(), str(e)[:200]))
    refuse("role capacity", role="capacity", frag="role baseline")
    refuse("role None (inferred capacity)", role=None, frag="role baseline")
    refuse("--arm s640", arm="s640", frag="m640 only")
    refuse("seeds 0,1", seeds="0,1", frag="one seed")
    refuse("seed 3", seed=3, seeds="3", exp="e2_w_m640_seed3", frag="E2's seeds")
    refuse("a wrong experiment name", exp="e2_w_m640_s0", frag="pre-registered")
    refuse("another manifest (train_core)", manifest=W.v2_dir() / "train_core.jsonl", frag="base_v2")
    refuse("--final-exams with test", final_exams=["dev", "imageweeds", "test"], frag="may not read test")
    refuse("dev only", final_exams=["dev"], frag="E2's finals")
    rp = C.INC_DIR / REF / "exp.json"
    with saved(rp):
        rp.unlink()
        refuse("b_v2_m640 missing", frag="has no exp.json")
    with saved(rp):
        x = json.loads(rp.read_text())
        x["base"]["recipe"]["epochs"] = 99
        rp.write_text(json.dumps(x))
        refuse("b_v2_m640's recipe changed (epochs 99): W refused", frag="key for key")
        e2_build("S", 0)
        okS = (C.INC_DIR / RC.e2_exp("S", 0) / "exp.json").is_file()
        cases.append(("  ... while S, which differs from it in the recipe anyway, is built", okS, ""))
        drop(RC.e2_exp("S", 0))
    with saved(rp):
        x = json.loads(rp.read_text())
        x["base"]["manifest_sha256"] = "0" * 64
        rp.write_text(json.dumps(x))
        refuse("b_v2_m640 built on another manifest", frag="base.manifest_sha256")
        refuse("  (S too)", letter="S", exp=RC.e2_exp("S", 0), frag="base.manifest_sha256")
    # every other key that defines training (e2_record's pairs), and the reference's seeds
    for what, change, frag in (
            ("b_v2_m640 built under another LOCK v2", lambda x: x["splits_v2"].update(lock_sha256="0" * 64),
             "splits_v2.lock_sha256"),
            ("b_v2_m640 built under another never-train index", lambda x: x["splits_v2"].update(
                nevertrain_sha256="0" * 64), "splits_v2.nevertrain_sha256"),
            ("b_v2_m640's arm record another batch", lambda x: x["arm"].update(batch=int(x["arm"].get("batch") or 16)
                                                                              + 1), "in arm"),
            ("b_v2_m640's arm record another checkpoint sha256", lambda x: x["arm"].update(
                **{k: "0" * 64 for k in x["arm"] if k.endswith("sha256")} or {"weights_sha256": "0" * 64}), "in arm"),
            ("b_v2_m640 with another image count", lambda x: x["base"].update(n_images=x["base"]["n_images"] + 1),
             "base.n_images"),
            ("b_v2_m640 under another protocol stamp", lambda x: x.update(protocol="v2"), "protocol"),
            ("b_v2_m640 deciding on another exam", lambda x: x.update(decision_exam="imageweeds"), "decision_exam"),
            ("b_v2_m640 of another type", lambda x: x.update(type="segment"), "type"),
            ("b_v2_m640 without seed 0 (seeds 1, 2)", lambda x: x.update(seeds=[1, 2]), "it has no seed 0")):
        with saved(rp):
            x = json.loads(rp.read_text())
            change(x)
            rp.write_text(json.dumps(x))
            refuse(what, frag=frag)
    vp = T.e1_verdict_path()
    with saved(vp):
        vp.unlink()
        refuse("e1_v1.json missing", frag="not a decided E1 verdict")
        write_e1_verdict(status="pending")
        refuse("e1_v1.json pending", frag="not a decided E1 verdict")
        write_e1_verdict(qualifies=False)
        refuse("e1_v1.json does not qualify E1-B", frag="does not qualify")
        write_e1_verdict(exp="e1_a_m640")
        refuse("e1_v1.json names another experiment (no E1-B there)", frag="is not E1's arm B")
        write_e1_verdict(summary_sha256="6" * 64)
        refuse("E1-B built from another splits v3 summary than E1's verdict was decided on", frag="splits v3 summary")
    bp = C.INC_DIR / E1B / "exp.json"
    with saved(bp):
        x = json.loads(bp.read_text())
        x["e1"]["arm"] = "A"
        bp.write_text(json.dumps(x))
        refuse("E1-B's exp.json says arm A", frag="is not E1's arm B")
    rj = C.INC_DIR / E1B / "runs" / "base__s0" / "run.json"
    w0 = C.INC_DIR / E1B / "runs" / "base__s0" / "weights" / "final.pt"
    with saved(rj):
        rj.write_text(json.dumps(dict(json.loads(rj.read_text()), status="running")))
        refuse("E1-B's base__s0 not done", frag="is not done")
    with saved(w0):
        w0.unlink()
        refuse("its final.pt missing", frag="is missing")
    with saved(w0):
        other = C.INC_DIR / E1B / "runs" / "base__s1" / "weights" / "final.pt"
        w0.unlink()
        os.symlink(str(other), str(w0))
        refuse("its final.pt a symlink", frag="symlink")
    with saved(w0):
        w0.write_bytes(w0.read_bytes() + b"x")
        refuse("its final.pt modified", frag="its run.json records")
    bad = [(w, d) for w, ok, d in cases if not ok]
    check("each refusal names its reason and leaves no directory of the experiment (%d cases)" % len(cases), not bad,
          bad)

    print("inc2.baseline build --e2: success")
    summ, defn, _res = e2_build("W", 0)
    d0 = json.loads((root / "exp.json").read_text())
    e2 = d0["e2"]
    rj0 = json.loads(rj.read_text())
    ref = json.loads(rp.read_text())
    check("E2-W seed 0: init_weights is the absolute path of E1-B's base__s0 final.pt, hashing as its run.json records",
          d0["init_weights"] == str(w0.resolve()) == e2["init"]["path"] and pathlib.Path(d0["init_weights"]).is_absolute()
          and e2["init"]["sha256"] == rj0["weights_sha256"] == W.sha(w0) and e2["init"]["run_id"] == "base__s0"
          and e2["init"]["exp"] == E1B and e2["seed"] == 0 and e2["arm"] == "W", e2.get("init"))
    check("the e2 record: the reference by its exp.json sha256 and manifest, what differs (init_weights only), what "
          "is declared, E1's verdict by sha256 and its decision keys, the training environment and E1-B's guard",
          e2["reference"]["exp_json_sha256"] == W.sha(rp) and e2["reference"]["manifest_sha256"]
          == ref["base"]["manifest_sha256"] == d0["base"]["manifest_sha256"]
          and e2["differs_from_reference"] == ["init_weights"] and "base.recipe" in e2["compared"]
          and e2["declared"]["final_exams"][1] == ["dev", "imageweeds"] and e2["declared"]["seeds"] == [[0, 1, 2], [0]]
          and e2["e1_verdict"]["sha256"] == W.sha(T.e1_verdict_path())
          and e2["e1_verdict"]["decision"] == {k: E1_DECISION.get(k) for k in RC.E2_E1_DECISION_KEYS}
          and e2["reference"]["training_env"] == local_env()
          and e2["init"]["guard"]["lock_sha256"] == lock_shas()[0] and e2["decided_by"] == RC.E2_DECIDED_BY, e2)
    check("finals dev and imageweeds, role baseline, the cold recipe (no recipe_name), research-only inherited from "
          "E1-B, priced from 681,100 image-epochs",
          d0["final_exams"] == ["dev", "imageweeds"] and d0["role"] == "baseline" and "recipe_name" not in d0
          and d0["base"]["recipe"] == RC.cold("m640") and d0["research_only"]["flag"] is True
          and d0["research_only"]["init_research_only"]["flag"] is True
          and d0["cost_estimate"]["image_epochs"] == 100 * d0["base"]["n_images"] and d0["seeds"] == [0]
          and summ["e2"] == e2, {k: d0.get(k) for k in ("final_exams", "role", "research_only")})
    sp = json.loads((root / "runs" / "base__s0" / "spec.json").read_text())
    check("the pinned driver's base__s0 spec starts from that path with seed 0 (driver unchanged)",
          sp["init"] == str(w0.resolve()) and sp["recipe"]["seed"] == 0 and sp["kind"] == "base", sp)
    e = refused(e2_build, "W", 0)
    check("a second build of it is refused", e is not None and "already built" in str(e), e)
    _summ, ds, _r = e2_build("S", 0)
    check("E2-S seed 0: recipe_name x1b, the table's x1b, differs in init_weights and base.recipe, priced from 340,550 "
          "image-epochs", ds["recipe_name"] == "x1b" and ds["base"]["recipe"] == RC.incremental("x1b", "m640")
          and ds["e2"]["differs_from_reference"] == ["init_weights", "base.recipe"]
          and ds["cost_estimate"]["image_epochs"] == 50 * ds["base"]["n_images"], ds.get("e2", {}).get("differs_from_reference"))
    check("research_only: base_v2's flag OR E1-B's, an unknown E1-B flag counting as true",
          B.e2_research_only({"flag": False, "basis": "x"}, {"init": {"exp": "e", "research_only": {"flag": True}}})["flag"]
          is True
          and B.e2_research_only({"flag": False, "basis": "x"}, {"init": {"exp": "e", "research_only": {"flag": False}}})[
              "flag"] is False
          and B.e2_research_only({"flag": False}, {"init": {"exp": "e", "research_only": {"flag": "unknown"}}})["flag"]
          is True and B.e2_research_only({"flag": False}, {"init": {"exp": "e"}})["flag"] is True
          and B.e2_research_only({"flag": True}, {"init": {"exp": "e", "research_only": {"flag": False}}})["flag"] is True)
    return d0


# ------------------------------------------------------------------ 5
def _prod(exp):
    """The experiment as a production one (exp.json testing false)."""
    p = C.INC_DIR / exp / "exp.json"
    x = json.loads(p.read_text())
    x["testing"] = False
    p.write_text(json.dumps(x))
    return x


def test_train_production():
    print("inc2.train in production: E2's base runs")
    import torch
    w0, s0 = RC.e2_exp("W", 0), RC.e2_exp("S", 0)
    dw, ds = _prod(w0), _prod(s0)
    init = dw["init_weights"]
    man = dw["base"]["manifest"]

    def go(exp, run_id, recipe, kind="base", init_=None, **kw):
        if kind != "final":
            kw.update(train_manifest=man, recipe=recipe)
        p = W.spec(run_id, kind, exp=exp, init=init_ or init, **kw)
        rc = W.run(p)
        return rc, W.run_json(p)
    if torch.cuda.is_available():
        print("  NOTE a CUDA device is present; the production device stage is not exercised")
        return
    rc, rj = go(w0, "base__s0", dict(RC.e2_recipe("W"), seed=0))
    check("E2-W base__s0 from the recorded init passes stage recipe and stops at device; run.json records e2, the "
          "init check and the environment check passed",
          rc == 1 and rj.get("stage") == "device" and rj.get("recipe_name") == "cold"
          and (rj.get("init_check") or {}).get("passed") is True and rj.get("e2", {}).get("arm") == "W"
          and rj["e2"].get("init_sha256") == dw["e2"]["init"]["sha256"]
          and (rj.get("training_env_check") or {}).get("passed") is True,
          (rj.get("stage"), (rj.get("error") or "")[-400:], rj.get("init_check")))
    rc, rj = go(s0, "base__s0", dict(RC.e2_recipe("S"), seed=0))
    check("E2-S base__s0 with x1b passes stage recipe (recipe_name x1b) and stops at device",
          rc == 1 and rj.get("stage") == "device" and rj.get("recipe_name") == "x1b",
          (rj.get("stage"), (rj.get("error") or "")[-400:]))
    out = {}
    out["E2-S with the cold recipe"] = go(s0, "base__s0", dict(RC.e2_recipe("W"), seed=0))
    out["E2-W with x1b"] = go(w0, "base__s0", dict(RC.e2_recipe("S"), seed=0))
    plain = C.INC_DIR / "e2t_plain"
    plain.mkdir(parents=True, exist_ok=True)
    (plain / "exp.json").write_text(json.dumps(dict({k: dw[k] for k in ("type", "seeds", "base")}, exp="e2t_plain",
                                                    recipe_name="x1b", testing=False,
                                                    **RC.stamp(RC.resolve_arm("m640", require_weights=False)))))
    out["x1b in an exp.json without an e2 record"] = go("e2t_plain", "base__s0", dict(RC.e2_recipe("S"), seed=0),
                                                         init_="yolo11m.pt")
    out["the arm's checkpoint as init"] = go(w0, "base__s0", dict(RC.e2_recipe("W"), seed=0), init_="yolo11m.pt")
    with saved(pathlib.Path(init)):
        pathlib.Path(init).write_bytes(pathlib.Path(init).read_bytes() + b"x")
        out["the init file changed after the build"] = go(w0, "base__s0", dict(RC.e2_recipe("W"), seed=0))
    out["spec seed 1 under e2 seed 0"] = go(w0, "base__s0", dict(RC.e2_recipe("W"), seed=1))
    rjp = C.INC_DIR / E1B / "runs" / "base__s0" / "run.json"
    with saved(rjp):
        rjp.write_text(json.dumps(dict(json.loads(rjp.read_text()), weights_sha256="1" * 64)))
        out["E1-B's run.json records other weights"] = go(w0, "base__s0", dict(RC.e2_recipe("W"), seed=0))
    vp = T.e1_verdict_path()
    with saved(vp):
        write_e1_verdict(qualifies=False)
        out["E1's verdict rewritten not qualifying"] = go(w0, "base__s0", dict(RC.e2_recipe("W"), seed=0))
    with saved(vp):
        write_e1_verdict(diff=0.03)
        out["E1's verdict with another D"] = go(w0, "base__s0", dict(RC.e2_recipe("W"), seed=0))
    ep = C.INC_DIR / w0 / "exp.json"
    with saved(ep):
        x = json.loads(ep.read_text())
        x["e2"]["reference"]["training_env"]["torch_version"] = "0.0.1"
        ep.write_text(json.dumps(x))
        out["another training environment than the reference's"] = go(w0, "base__s0", dict(RC.e2_recipe("W"), seed=0))
    out["a union run under an E2 exp.json"] = go(w0, "union__s0", dict(RC.e2_recipe("W"), seed=0), kind="union")
    with saved(ep):
        x = json.loads(ep.read_text())
        x["seeds"] = [0, 1]
        ep.write_text(json.dumps(x))
        out["an exp.json listing two seeds under e2 seed 0"] = go(w0, "base__s0", dict(RC.e2_recipe("W"), seed=0))
    with saved(ep):
        x = json.loads(ep.read_text())
        x["exp"] = "e2_w_m640_seed9"
        ep.write_text(json.dumps(x))
        out["an exp.json not named as E2-W seed 0 is"] = go(w0, "base__s0", dict(RC.e2_recipe("W"), seed=0))
    bad = {k: (rj_.get("stage"), (rj_.get("error") or "")[-200:]) for k, (rc_, rj_) in out.items()
           if not (rc_ == 1 and rj_.get("stage") == "recipe")}
    check("refused at stage recipe: %s" % ", ".join(out), not bad, bad)
    check("  the messages name E2's init and environment",
          "E1-B's recorded weights" in (out["the arm's checkpoint as init"][1].get("error") or "")
          and "environment" in (out["another training environment than the reference's"][1].get("error") or ""))
    with saved(vp):
        write_e1_verdict()
        x = json.loads(vp.read_text())
        x["generated_utc"] = "2026-10-05T12:00:00Z"
        vp.write_text(json.dumps(x))
        rc, rj = go(w0, "base__s0", dict(RC.e2_recipe("W"), seed=0))
    check("E1's verdict rewritten with only generated_utc changed is not refused (stops at device)",
          rc == 1 and rj.get("stage") == "device", (rj.get("stage"), (rj.get("error") or "")[-300:]))
    # experiment_arm: init_weights other than the arm's checkpoint only when an e2 record names it
    (plain / "exp.json").write_text(json.dumps(dict(RC.stamp(RC.resolve_arm("m640", require_weights=False)),
                                                    exp="e2t_plain", type="baseline", init_weights=init)))
    e1 = refused(T.experiment_arm, "e2t_plain")
    with saved(ep):
        x = json.loads(ep.read_text())
        x["init_weights"] = str(C.INC_DIR / E1B / "runs" / "base__s1" / "weights" / "final.pt")
        ep.write_text(json.dumps(x))
        e2_ = refused(T.experiment_arm, w0)
    check("experiment_arm: another init_weights without an e2 record is refused (as before), and one the e2 record "
          "does not name", e1 is not None and "no e2 record names it" in str(e1) and e2_ is not None, (e1, e2_))
    rc, rj = go(w0, "final__base__s0", None, kind="final", exams=["dev"])
    check("a final run under an E2 exp.json passes stage recipe (its init is the base run's weights)",
          rc == 1 and rj.get("stage") == "device", (rj.get("stage"), (rj.get("error") or "")[-300:]))
    for p in (C.INC_DIR / w0 / "runs").glob("*/run.json"):
        p.unlink()
    for exp in (w0, s0):
        x = json.loads((C.INC_DIR / exp / "exp.json").read_text())
        x["testing"] = W.TESTING
        (C.INC_DIR / exp / "exp.json").write_text(json.dumps(x))
    shutil.rmtree(str(plain), ignore_errors=True)


# ------------------------------------------------------------------ 6, 7
def test_whole_load():
    print("the whole load (init_transfer)")
    import torch
    p = C.INC_DIR / E1B / "runs" / "base__s0" / "weights" / "final.pt"
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    m = copy.deepcopy(ck["model"]).float()
    r = T.init_transfer(m, p)
    check("the init's own model: every tensor equal (whole)", r["whole"] and r["equal"] == r["tensors"] > 0
          and not r["first_differing"], r)

    class Wrap(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner
    check("a DDP-style wrapper is unwrapped (state_dict names are the checkpoint's)", T.init_transfer(Wrap(m), p)["whole"])
    bn = next(x for x in m.modules() if isinstance(x, torch.nn.BatchNorm2d))
    with torch.no_grad():
        bn.running_mean.add_(1.0)
    r2 = T.init_transfer(m, p)
    check("a BatchNorm running_mean that differs counts (the state_dict, not only the parameters)",
          not r2["whole"] and r2["n_differing"] == 1, r2)
    with torch.no_grad():
        bn.running_mean.sub_(1.0)
        next(m.parameters()).add_(0.5)
    r3 = T.init_transfer(m, p)
    e = refused(T.require_whole_load, r3, p, testing=False)
    check("a perturbed parameter: not whole; require_whole_load refuses in production (stage train) and records in "
          "testing", not r3["whole"] and r3["n_differing"] == 1 and isinstance(e, T.RunError) and e.stage == "train"
          and refused(T.require_whole_load, r3, p, testing=True) is None
          and refused(T.require_whole_load, r, p, testing=False) is None, (r3, e))

    print("the whole load on a real CPU run (1 epoch from E1-B's weights, test mode)")
    exp = RC.e2_exp("W", 0)
    man = json.loads((C.INC_DIR / exp / "exp.json").read_text())["base"]["manifest"]
    t0 = time.time()
    sp = W.spec("base__s0", "base", exp=exp, init=str(p.resolve()), train_manifest=man, recipe=W.recipe(seed=0))
    rc = W.run(sp)
    rj = W.run_json(sp)
    it = rj.get("init_transfer") or {}
    check("the run is done (%.0fs); init_transfer: every tensor of the trained model equal to the init's at setup, "
          "the init check and the environment check passed" % (time.time() - t0),
          rc == 0 and rj.get("status") == "done" and it.get("whole") is True and it.get("equal") == it.get("tensors") > 0
          and (rj.get("init_check") or {}).get("passed") is True
          and (rj.get("training_env_check") or {}).get("passed") is True and rj.get("e2", {}).get("seed") == 0,
          (rc, rj.get("stage"), (rj.get("error") or "")[-600:], it))
    # an init whose class head is 12-class: Ultralytics skips the head's last convs (intersect_dicts by shape)
    w1 = e1b_run(1, nc=12)
    exp1 = RC.e2_exp("W", 1)
    e2_build("W", 1)
    sp1 = W.spec("base__s1", "base", exp=exp1, init=str(w1.resolve()), train_manifest=man, recipe=W.recipe(seed=1))
    rc1 = W.run(sp1)
    rj1 = W.run_json(sp1)
    it1 = rj1.get("init_transfer") or {}
    check("an init with a 12-class head: the differing head tensors are recorded (a testing run goes on)",
          rc1 == 0 and it1.get("whole") is False and it1.get("n_differing", 0) >= 1
          and any(".cv3." in n for n in it1.get("first_differing") or []), (rc1, (rj1.get("error") or "")[-400:], it1))
    drop(exp1)
    e1b_run(1)


# ------------------------------------------------------------------ 8
def test_bootstrap():
    print("the paired bootstrap under inc2/e2/species_se")
    ref = N.synth_arrays(seed=1)
    other = [N.synth_arrays(seed=10 + i) for i in range(3)]
    b1 = B.native_bootstrap(other, [ref, ref, ref], resamples=40, seed_text=B.E2_SEED_TEXT)
    b2 = B.native_bootstrap(other, [ref, ref, ref], resamples=40, seed_text=B.E2_SEED_TEXT)
    ind = N.independent_se(other, [ref, ref, ref], 40, B.E2_SEED_TEXT)
    nat = B.native_bootstrap(other, [ref, ref, ref], resamples=40)
    check("deterministic, equal to an independent recomputation (%.6f vs %.6f), and another draw than the native "
          "rule's seed text" % (b1["se"], ind), b1 == b2 and abs(b1["se"] - ind) < 1e-9 and nat["se"] != b1["se"],
          (b1["se"], ind, nat["se"]))


# ------------------------------------------------------------------ 9, 10: the rule on synthetic native files
RULE_REF = "e2v_ref"
REF_MAN = "f" * 64


def _e2_exp_doc(exp, letter, seed, init_sha="1" * 64, ref_man=REF_MAN, decision=None, testing=CPU32):
    d = {"exp": exp, "type": "baseline", "role": "baseline", "seeds": [seed], "final_exams": ["dev", "imageweeds"],
         "testing": testing, "base": {"manifest_sha256": ref_man},
         "e2": {"arm": letter, "seed": seed,
                "init": {"exp": E1B, "run_id": "base__s%d" % seed, "sha256": init_sha},
                "e1_verdict": {"decision": decision or {k: E1_DECISION.get(k) for k in RC.E2_E1_DECISION_KEYS}},
                "reference": {"manifest_sha256": ref_man, "training_env": local_env()}}}
    d.update(RC.stamp(RC.resolve_arm("m640", require_weights=False)))
    return d


def _native_doc(exp, rid, val, arr, wsha, production=True, compared=True, settings=None, pc=None, ag=None,
                ultra="8.4.37"):
    st = dict({"imgsz": 640, "batch": 32, "conf": 0.001, "iou": 0.7, "half": True, "rect": True, "max_det": 300,
               "image_correct_conf": 0.25, "image_correct_iou": 0.5}, **(settings or {}))
    return {"format": SN.FORMAT, "exam": "dev", "imgsz": 640, "exp": exp, "run_id": rid, "species_map50_95": val,
            "agnostic_map50_95": ag if ag is not None else val + 0.01,
            "per_class": pc or {n: val for n in SC.SPECIES}, "native_production": production,
            "other_deviations": [] if production else ["fp32 on device cpu"], "manifest_sha256": "m" * 64,
            "key_order_sha256": C.sha256_text("\n".join(str(k) for k in arr["keys"])),
            "n_images": len(arr["keys"]), "locked_scorer_sha256": "s" * 64, "ultralytics_version": ultra,
            "settings": st, "weights_sha256": wsha,
            "vs_protocol_score": {"compared": compared}, "protocol_score": {"production": True, "file": "dev.json"}}


def fake_run_files(exp, seed, val, arr, wsha, base=True, recipe_name="cold", init_sha="1" * 64, skip=False, **nat):
    """runs/base__s<seed> (an E2 base run's record) and runs/final__base__s<seed> with its native dev@640 file."""
    rd = C.INC_DIR / exp / "runs"
    if base:
        (rd / ("base__s%d" % seed)).mkdir(parents=True, exist_ok=True)
        (rd / ("base__s%d" % seed) / "run.json").write_text(json.dumps(dict({
            "status": "done", "init_sha256": init_sha, "recipe_name": recipe_name, "weights_sha256": wsha,
            "testing": False, "protocol_recipe": True, "init_check": {"passed": True},
            "training_env_check": {"passed": True}, "init_transfer": {"whole": True, "tensors": 10, "equal": 10}},
            **local_env())))
    fd = rd / ("final__base__s%d" % seed)
    (fd / "scores").mkdir(parents=True, exist_ok=True)
    (fd / "run.json").write_text(json.dumps({"status": "done", "weights_sha256": wsha}))
    if skip:
        return
    js, npz = SN.paths_for(fd / "scores", "dev", 640)
    sha = SC.save_npz(npz, arr)
    doc = _native_doc(exp, "final__base__s%d" % seed, val, arr, wsha, **nat)
    doc["images"] = {"path": str(npz), "sha256": sha}
    js.write_text(json.dumps(doc))


def rule_world(tag, wv, sv, rv=(0.850, 0.851, 0.849), warr=None, sarr=None, rarr=None, skip=(), w_over=None,
               s_over=None):
    """Six E2 experiments (W and S of seeds 0-2) and a reference with synthetic native files: {'W': [...], 'S': [...]}."""
    ref = N.synth_arrays(seed=1)
    for e in [x for x in os.listdir(str(C.INC_DIR)) if x.startswith(tag + "_") or x == RULE_REF]:
        shutil.rmtree(str(C.INC_DIR / e))
    rroot = C.INC_DIR / RULE_REF
    rroot.mkdir(parents=True)
    rdoc = {"exp": RULE_REF, "type": "baseline", "seeds": [0, 1, 2], "final_exams": ["dev", "imageweeds", "test"],
            "testing": CPU32, "base": {"manifest_sha256": REF_MAN}}
    rdoc.update(RC.stamp(RC.resolve_arm("m640", require_weights=False)))
    (rroot / "exp.json").write_text(json.dumps(rdoc))
    for s in (0, 1, 2):
        fake_run_files(RULE_REF, s, rv[s], (rarr or [ref] * 3)[s], "r%d" % s * 32, base=False)
        (rroot / "runs" / ("base__s%d" % s)).mkdir(parents=True, exist_ok=True)
        (rroot / "runs" / ("base__s%d" % s) / "run.json").write_text(json.dumps(dict({"status": "done"},
                                                                                     **local_env())))
    exps = {"W": [], "S": []}
    for letter, vals, arrs, over in (("W", wv, warr, w_over), ("S", sv, sarr, s_over)):
        for s in (0, 1, 2):
            e = "%s_%s%d" % (tag, letter.lower(), s)
            (C.INC_DIR / e).mkdir(parents=True)
            (C.INC_DIR / e / "exp.json").write_text(json.dumps(_e2_exp_doc(e, letter, s)))
            kw = dict((over or {}).get(s) or {})
            fake_run_files(e, s, vals[s], (arrs or [ref] * 3)[s], "%s%d" % (letter.lower(), s) * 32,
                           recipe_name=RC.E2_ARMS[letter], skip=(letter, s) in skip, **kw)
            exps[letter].append(e)
    return exps


def test_rule():
    print("E2's rule on synthetic native files")
    exps = rule_world("e2v", [0.861, 0.862, 0.860], [0.851, 0.852, 0.850])
    d = B.e2_decision(exps, reference=RULE_REF, resamples=40)
    a = d["arms"]["W"]
    check("W qualifies with both conditions (D %.4f > 2 pooled sd %.4f, > SE %.4f); S does not (D 0.001); chosen W"
          % (a["diff"], a["two_pooled_sd"], a["se_diff"]),
          d["status"] == "decided" and a["qualifies"] and all(a["conditions"].values()) and abs(a["diff"] - 0.011) < 1e-9
          and a["se_diff"] == 0.0 and not d["arms"]["S"]["qualifies"] and d["qualifying"] == ["W"]
          and d["chosen"] == "W", (a.get("conditions"), d["qualifying"], d["chosen"]))
    sa, sr = statistics.stdev(a["dev"]), statistics.stdev(a["reference_dev"])
    check("pooled sd is sqrt((sd_arm^2 + sd_ref^2) / 2) of the sample sds",
          abs(a["pooled_sd"] - math.sqrt((sa ** 2 + sr ** 2) / 2.0)) < 1e-12 and a["two_pooled_sd"] == 2 * a["pooled_sd"])
    rep = a["reported"]
    check("reported beside, not deciding: agnostic dev, Carpetweed / SpottedSpurge / Purslane with their SE, the "
          "protocol dev means, the training environment",
          sorted(rep["species"]) == sorted(B.E2_REPORTED_SPECIES) and rep["agnostic"]["diff"] is not None
          and all(v["se"] is not None for v in rep["species"].values()) and "protocol_dev" in rep
          and rep["training_env"]["same"] is True, rep)
    old = B.native_bootstrap
    B.native_bootstrap = lambda *x, **k: {"se": 1.0, "n_valid": 40, "per_species": {}}
    try:
        d2 = B.e2_decision(exps, reference=RULE_REF, resamples=40)
    finally:
        B.native_bootstrap = old
    c = d2["arms"]["W"]["conditions"]
    check("only D > 2 pooled sd (SE 1.0): W does not qualify", c == {"above_2_pooled_sd": True, "above_se": False}
          and not d2["arms"]["W"]["qualifies"] and d2["chosen"] is None, c)
    exps = rule_world("e2v", [0.880, 0.845, 0.865], [0.851, 0.852, 0.850])
    d3 = B.e2_decision(exps, reference=RULE_REF, resamples=40)
    c = d3["arms"]["W"]["conditions"]
    check("only D > SE (seed sd large, D %.4f under 2 pooled sd %.4f): W does not qualify"
          % (d3["arms"]["W"]["diff"], d3["arms"]["W"]["two_pooled_sd"]),
          c == {"above_2_pooled_sd": False, "above_se": True} and not d3["arms"]["W"]["qualifies"], c)
    exps = rule_world("e2v", [0.861, 0.862, 0.860], [0.871, 0.872, 0.870])
    d4 = B.e2_decision(exps, reference=RULE_REF, resamples=20)
    check("both qualify: the larger D is chosen (S)", d4["qualifying"] == ["W", "S"] and d4["chosen"] == "S",
          (d4["qualifying"], d4["chosen"]))
    exps = rule_world("e2v", [0.871, 0.872, 0.870], [0.861, 0.862, 0.860])
    d5 = B.e2_decision(exps, reference=RULE_REF, resamples=20)
    exps = rule_world("e2v", [0.861, 0.862, 0.860], [0.861, 0.862, 0.860])
    d6 = B.e2_decision(exps, reference=RULE_REF, resamples=20)
    check("... W when its D is larger; a tie goes to S (the shorter schedule)",
          d5["chosen"] == "W" and d6["qualifying"] == ["W", "S"] and d6["chosen"] == "S", (d5["chosen"], d6["chosen"]))
    exps = rule_world("e2v", [0.861, 0.862, 0.860], [0.861, 0.862, 0.860], skip=(("S", 2),))
    d7 = B.e2_decision(exps, reference=RULE_REF, resamples=20)
    check("one native file missing: pending, nothing qualifying, no choice",
          d7["status"] == "pending" and d7["qualifying"] == ["W"] and d7["chosen"] is None
          and d7["arms"]["S"]["missing"] == ["e2v_s2/final__base__s2"], (d7["status"], d7["arms"]["S"]))
    # every refusal on the production path
    cases = {}

    def refuse(name, frag, **kw):
        ex = rule_world("e2v", [0.861, 0.862, 0.860], [0.851, 0.852, 0.850], **kw)
        e = refused(B.e2_decision, ex, reference=RULE_REF, resamples=10)
        cases[name] = (e is not None and frag in str(e), str(e)[:200])
        return ex
    refuse("a stamp mismatch (Ultralytics' version)", "ultralytics_version", w_over={1: {"ultra": "8.4.38"}})
    refuse("a test-mode native file in production", "test-mode", w_over={0: {"production": False}})
    refuse("a native file not checked against its protocol score", "protocol score", s_over={2: {"compared": False}})
    ex = rule_world("e2v", [0.861, 0.862, 0.860], [0.851, 0.852, 0.850])
    p = C.INC_DIR / "e2v_w1" / "runs" / "final__base__s1" / "scores" / "dev@640.json"
    x = json.loads(p.read_text())
    x["weights_sha256"] = "9" * 64
    p.write_text(json.dumps(x))
    e = refused(B.e2_decision, ex, reference=RULE_REF, resamples=10)
    cases["a native score naming other weights than its base run's"] = (e is not None and "names weights" in str(e),
                                                                         str(e)[:200])
    ex = rule_world("e2v", [0.861, 0.862, 0.860], [0.851, 0.852, 0.850])
    p = C.INC_DIR / "e2v_s0" / "runs" / "base__s0" / "run.json"
    p.write_text(json.dumps(dict(json.loads(p.read_text()), init_sha256="7" * 64)))
    e = refused(B.e2_decision, ex, reference=RULE_REF, resamples=10)
    cases["a base run that did not start from the recorded init"] = (e is not None and "recorded init" in str(e),
                                                                     str(e)[:200])
    ex = rule_world("e2v", [0.861, 0.862, 0.860], [0.851, 0.852, 0.850])
    p = C.INC_DIR / "e2v_w2" / "runs" / "base__s2" / "run.json"
    p.write_text(json.dumps(dict(json.loads(p.read_text()), init_transfer={"whole": False, "tensors": 10, "equal": 8})))
    e = refused(B.e2_decision, ex, reference=RULE_REF, resamples=10)
    cases["a base run whose init did not load whole"] = (e is not None and "loaded whole" in str(e), str(e)[:200])
    ex = rule_world("e2v", [0.861, 0.862, 0.860], [0.851, 0.852, 0.850])
    e = refused(B.e2_decision, {"W": [ex["W"][0], ex["W"][0], ex["W"][1]], "S": ex["S"]}, reference=RULE_REF,
                resamples=10)
    cases["two experiments of one arm with one seed"] = (e is not None and "two experiments" in str(e), str(e)[:200])
    ex = rule_world("e2v", [0.861, 0.862, 0.860], [0.851, 0.852, 0.850])
    p = C.INC_DIR / "e2v_s1" / "exp.json"
    x = json.loads(p.read_text())
    x["e2"]["reference"]["manifest_sha256"] = "0" * 64
    p.write_text(json.dumps(x))
    e = refused(B.e2_decision, ex, reference=RULE_REF, resamples=10)
    cases["an experiment built against another reference manifest"] = (e is not None and "another manifest" in str(e),
                                                                        str(e)[:200])
    ex = rule_world("e2v", [0.861, 0.862, 0.860], [0.851, 0.852, 0.850])
    p = C.INC_DIR / "e2v_s1" / "exp.json"
    x = json.loads(p.read_text())
    x["e2"]["init"]["exp"] = "e1_b_other"
    p.write_text(json.dumps(x))
    e = refused(B.e2_decision, ex, reference=RULE_REF, resamples=10)
    cases["experiments starting from different E1-B records"] = (e is not None and "different E1-B" in str(e),
                                                                 str(e)[:200])
    bad = {k: v[1] for k, v in cases.items() if not v[0]}
    check("refused: %s" % "; ".join(cases), not bad, bad)
    # the decision file: dev only
    ex = rule_world("e2v", [0.861, 0.862, 0.860], [0.851, 0.852, 0.850])
    out = C.INC_DIR / "cap_e2_rule"
    dd, rep = B.e2_verdict(ex, reference=RULE_REF, out_dir=out, resamples=20)
    doc = json.loads((out / "e2_v1.json").read_text())
    vals = list(N.walk_values(doc))
    check("capacity/e2_v1.json: dev only (no imageweeds or test key or value, no score path; the autopilot's scrub "
          "drops nothing), the rule and the bootstrap recorded; the report (people) holds ImageWeeds",
          not BP.dev_leaks(doc) and not E.scrub(doc)[1] and not E.leaks(doc)
          and not [v for v in vals if v in ("test", "imageweeds")] and doc["rule"] == B.E2_RULE
          and doc["bootstrap"]["seed_text"] == "inc2/e2/species_se" and doc["bootstrap"]["resamples"] == 20
          and doc["exam"] == "dev" and "imageweeds" in rep["arms"]["W"]["exams"]
          and (out / "e2_v1_report.md").is_file(), (BP.dev_leaks(doc), E.scrub(doc)[1]))
    return ex, out


def test_verdict_file(ex, out):
    print("e2_verdict: a decided file is kept byte for byte, never rewritten")
    vp = out / "e2_v1.json"
    sha0 = W.sha(vp)

    def again():
        try:
            return B.e2_verdict(ex, reference=RULE_REF, out_dir=out, resamples=20)[0], None
        except B.BaselineError as e:
            return {}, e
    d, err = again()
    check("a second e2_verdict keeps e2_v1.json byte for byte", err is None and d.get("kept") is True
          and W.sha(vp) == sha0, err)
    pd = C.INC_DIR / "e2v_w0" / "runs" / "base__s0" / "scores"
    pd.mkdir(parents=True, exist_ok=True)
    (pd / "dev.json").write_text(json.dumps({"exam": "dev", "production": True, "species_map50_95": 0.7}))
    d, err = again()
    check("  also when a field reported beside it changed (a base run's protocol dev score): kept",
          err is None and d.get("kept") is True and W.sha(vp) == sha0
          and d["arms"]["W"]["reported"]["protocol_dev"]["arm"]["n"] == 1, err)
    for name, kw in (("with 10 resamples", {"resamples": 10}), ("admitting test-mode files", {"testing_ok": True})):
        e = refused(B.e2_verdict, ex, reference=RULE_REF, out_dir=out, **dict({"resamples": 20}, **kw))
        check("a recomputation %s refuses (the parameters are part of the decision), the file unchanged" % name,
              e is not None and "never rewritten" in str(e) and W.sha(vp) == sha0, e)
    p = C.INC_DIR / "e2v_w1" / "runs" / "final__base__s1" / "scores" / "dev@640.json"
    x = json.loads(p.read_text())
    x["species_map50_95"] = 0.80
    p.write_text(json.dumps(x))
    e = refused(B.e2_verdict, ex, reference=RULE_REF, out_dir=out, resamples=20)
    check("a recomputation whose decision differs refuses, the file unchanged",
          e is not None and "never rewritten" in str(e) and W.sha(vp) == sha0, e)
    pend = rule_world("e2v", [0.861, 0.862, 0.860], [0.851, 0.852, 0.850], skip=(("W", 0),))
    out2 = C.INC_DIR / "cap_e2_pend"
    B.e2_verdict(pend, reference=RULE_REF, out_dir=out2, resamples=10)
    s1 = json.loads((out2 / "e2_v1.json").read_text())["status"]
    full = rule_world("e2v", [0.861, 0.862, 0.860], [0.851, 0.852, 0.850])
    B.e2_verdict(full, reference=RULE_REF, out_dir=out2, resamples=10)
    s2 = json.loads((out2 / "e2_v1.json").read_text())["status"]
    check("a pending file is overwritten by a decided one", (s1, s2) == ("pending", "decided"), (s1, s2))


# ------------------------------------------------------------------ 11: rescore-e2 end to end
def real_e2(exp, letter, seed, wseed, done=True, testing=CPU32):
    """An E2 experiment as built and run: exp.json (its e2 record), the base run's weights (a real checkpoint) and
    run.json, the final run (final.pt a symlink to them) with its protocol dev score at 640."""
    drop(exp)
    root = C.INC_DIR / exp
    root.mkdir(parents=True)
    doc = _e2_exp_doc(exp, letter, seed, ref_man=REF_MAN, testing=testing)
    (root / "exp.json").write_text(json.dumps(doc))
    rd = N.make_final(exp, seed, wseed, protocol=True, done=done)
    w = root / "runs" / ("base__s%d" % seed) / "weights" / "final.pt"
    (root / "runs" / ("base__s%d" % seed) / "run.json").write_text(json.dumps(dict({
        "status": "done", "init_sha256": "1" * 64, "recipe_name": RC.E2_ARMS[letter], "weights_sha256": W.sha(w),
        "testing": True}, **local_env())))
    return rd


def test_rescore():
    print("rescore-e2 end to end (real CPU passes, test mode)")
    ref = "e2r_ref"
    drop(ref)
    rroot = C.INC_DIR / ref
    rroot.mkdir(parents=True)
    rdoc = {"exp": ref, "type": "baseline", "seeds": [0, 1, 2], "final_exams": ["dev", "imageweeds", "test"],
            "testing": CPU32, "base": {"manifest_sha256": REF_MAN}}
    rdoc.update(RC.stamp(RC.resolve_arm("m640", require_weights=False)))
    (rroot / "exp.json").write_text(json.dumps(rdoc))
    for s in (0, 1, 2):
        N.make_final(ref, s, 70 + s)
    exps = {"W": [], "S": []}
    for letter in ("W", "S"):
        for s in (0, 1, 2):
            e = "e2r_%s%d" % (letter.lower(), s)
            real_e2(e, letter, s, 80 + s + (10 if letter == "S" else 0))
            exps[letter].append(e)
    out = C.INC_DIR / "cap_e2_rescore"
    # an undone final run refuses before anything is scored
    fj = C.INC_DIR / "e2r_s2" / "runs" / "final__base__s2" / "run.json"
    with saved(fj):
        fj.write_text(json.dumps(dict(json.loads(fj.read_text()), status="running")))
        e = refused(B.rescore_e2, exps, reference=ref, out_dir=out, resamples=20)
        written = [str(p) for p in C.INC_DIR.glob("e2r_*/runs/*/scores/*@640.json")]
    check("an undone final run refuses before anything is written",
          e is not None and "not done" in str(e) and not written and not out.exists(), (e, written))
    t0 = time.time()
    rec = B.rescore_e2(exps, reference=ref, out_dir=out, resamples=20)
    doc = json.loads((out / "e2_rescore.json").read_text())
    v = out / "e2_v1.json"
    check("rescore-e2 (%.0fs): every E2 final run scored on dev at 640 (written) and the reference's three, then "
          "the verdict (decided) and capacity/e2_rescore.json (complete, names and sha256s, no path)" % (time.time() - t0),
          rec["status"] == "complete" and doc == json.loads(json.dumps(rec))
          and all(r["status"] == "written" and r["score"] == "dev@640.json" and r["sha256"] for k in ("W", "S")
                  for r in doc["arms"][k]) and len(doc["arms"]["W"]) == len(doc["arms"]["S"]) == 3
          and [r["status"] for r in doc["reference"]["scores"]] == ["written"] * 3
          and doc["verdict"]["sha256"] == W.sha(v) and doc["verdict"]["status"] == "decided"
          and "/" not in json.dumps({k: doc[k] for k in ("arms", "reference", "verdict")}), doc)
    v0 = W.sha(v)
    rec2 = B.rescore_e2(exps, reference=ref, out_dir=out, resamples=20)
    check("a second run scores nothing (every file kept) and keeps e2_v1.json byte for byte",
          all(r["status"] == "kept" for k in ("W", "S") for r in rec2["arms"][k])
          and all(r["status"] == "kept" for r in rec2["reference"]["scores"]) and W.sha(v) == v0
          and rec2["verdict"]["sha256"] == v0)
    nat = json.loads((C.INC_DIR / "e2r_w0" / "runs" / "final__base__s0" / "scores" / "dev@640.json").read_text())
    check("each native file reproduced the run's protocol dev score (vs_protocol_score compared)",
          (nat.get("vs_protocol_score") or {}).get("compared") is True, nat.get("vs_protocol_score"))


# ------------------------------------------------------------------ 12, 13: the test read
READ_REF = "e2r_ref"            # test_rescore's reference (its final runs), the reference of the read worlds' verdicts


def read_world(tag, qualifying=("W",), chosen="W", **over):
    """Three E2-W and three E2-S experiments with base weights (bytes), and a decided verdict naming them, decided
    under the pre-registered parameters (E2_RULE, the reference READ_REF, the bootstrap's seed text and 1,000
    resamples, no test-mode file) unless `over` replaces a field."""
    ex = {"W": [], "S": []}
    arms = {}
    for letter in ("W", "S"):
        inputs = []
        for s in (0, 1, 2):
            e = "%s_%s%d" % (tag, letter.lower(), s)
            drop(e)
            (C.INC_DIR / e).mkdir(parents=True)
            (C.INC_DIR / e / "exp.json").write_text(json.dumps(_e2_exp_doc(e, letter, s)))
            w = C.INC_DIR / e / "runs" / ("base__s%d" % s) / "weights" / "final.pt"
            w.parent.mkdir(parents=True)
            w.write_bytes(os.urandom(256))
            inputs.append({"exp": e, "run_id": "final__base__s%d" % s, "sha256": "n" * 64, "images_sha256": "i" * 64,
                           "weights_sha256": W.sha(w), "init_sha256": "1" * 64})
            ex[letter].append(e)
        arms[letter] = {"status": "decided", "seeds": [0, 1, 2], "diff": 0.012 if letter in qualifying else 0.001,
                        "qualifies": letter in qualifying, "inputs": inputs}
    vp = C.INC_DIR / ("cap_%s" % tag) / "e2_v1.json"
    vp.parent.mkdir(parents=True, exist_ok=True)
    doc = {"format": B.E2_FORMAT, "status": "decided", "arms": arms, "qualifying": list(qualifying), "chosen": chosen,
           "rule": B.E2_RULE, "reference": {"exp": READ_REF, "seeds": [0, 1, 2]}, "testing_allowed": False,
           "bootstrap": {"seed_text": B.E2_SEED_TEXT, "resamples": B.E2_RESAMPLES}}
    doc.update(over)
    vp.write_text(json.dumps(doc))
    return ex, vp


def no_specs(ex, letter):
    return not [p for e in ex[letter] for p in (C.INC_DIR / e).glob("runs/e2test__s*/spec.json")] \
        and not [e for e in ex[letter] if (C.INC_DIR / e / B.E2_TEST_RECORD).exists()]


def _test_score(exp, s, val, wsha, stamps=None, **over):
    """The protocol test score of <exp>/runs/e2test__s<s> (inc2.train writes it): production, its weights."""
    p = C.INC_DIR / exp / "runs" / ("e2test__s%d" % s) / "scores" / "test.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = dict({"exam": "test", "production": True, "species_map50_95": val, "agnostic_map50_95": val + 0.01,
                "weights_sha256": wsha}, **(stamps or TEST_STAMPS))
    doc.update(over)
    p.write_text(json.dumps(doc))
    return p


TEST_STAMPS = {"scorer_sha256": "c" * 64, "manifest_sha256": "t" * 64, "key_order_sha256": "k" * 64}


def test_test_read():
    print("e2-test-read: each qualifying arm, once, after the verdict")
    ex, vp = read_world("e2t", qualifying=("W", "S"), chosen="W")
    missing = vp.with_name("none.json")
    out = {}
    out["before the verdict"] = (refused(B.e2_test_read, "W", verdict_path=missing, reference=READ_REF),
                                 "not a decided E2 verdict")
    out["the arm letter X"] = (refused(B.e2_test_read, "X", verdict_path=vp, reference=READ_REF),
                               "is not one of E2's arms")
    ex1, vp1 = read_world("e2u", qualifying=("W",), chosen="W")
    out["an arm that did not qualify"] = (refused(B.e2_test_read, "S", verdict_path=vp1, reference=READ_REF),
                                          "did not qualify")
    # a verdict decided under other parameters than the pre-registered ones opens no sealed test
    for name, over in (("testing_allowed true", {"testing_allowed": True}),
                       ("testing_allowed missing", {"testing_allowed": None}),
                       ("200 resamples", {"bootstrap": {"seed_text": B.E2_SEED_TEXT, "resamples": 200}}),
                       ("another seed text", {"bootstrap": {"seed_text": "inc2/e2/other", "resamples": 1000}}),
                       ("another rule", {"rule": B.E2_RULE + " (edited)"}),
                       ("another reference", {"reference": {"exp": "b_v2_s640"}})):
        exo, vpo = read_world("e2o", qualifying=("W",), chosen="W", **over)
        out["a verdict with %s" % name] = (refused(B.e2_test_read, "W", verdict_path=vpo, reference=READ_REF),
                                           "pre-registered parameters")
        if not no_specs(exo, "W"):
            out["a verdict with %s (nothing written)" % name] = (None, "")
    bad = {k: str(e)[:200] for k, (e, frag) in out.items() if e is None or frag not in str(e)}
    check("refused: %s; no spec written" % ", ".join(out), not bad and no_specs(ex, "W") and no_specs(ex, "S"), bad)
    res = B.e2_test_read("W", verdict_path=vp, reference=READ_REF)
    specs = [json.loads(pathlib.Path(r["specs"][0]).read_text()) for r in res.values()]
    for sp in specs:
        T.validate_spec(sp, pathlib.Path(sp["out_dir"]) / "spec.json")
    check("for the chosen arm: three kind-final specs on test from the base weights (the v2 executor accepts each), "
          "three run_inc2_job.sh argvs, e2_test_read.json with the verdict's sha256",
          sorted(res) == sorted(ex["W"]) and len(specs) == 3
          and all(sp["kind"] == "final" and sp["exams"] == ["test"] and sp["init"].endswith("weights/final.pt")
                  and sp["run_id"].startswith("e2test__s") for sp in specs)
          and all(r["argv"][0] == "sbatch" and r["argv"][-3].endswith("run_inc2_job.sh")
                  and r["argv"][3] == "--job-name=inc_%s_e2test" % e for e, r in res.items())
          and all(json.loads((C.INC_DIR / e / B.E2_TEST_RECORD).read_text())["verdict_sha256"] == W.sha(vp)
                  for e in ex["W"]), res)
    try:
        res_s = B.e2_test_read("S", verdict_path=vp, reference=READ_REF)
    except B.BaselineError as x:
        res_s = {"refused": {"arm": None, "chosen": None, "why": str(x)}}
    check("a qualifying arm the verdict did not choose (S) is read too, once (pre-registered: once per qualifying "
          "arm), its record naming the choice",
          sorted(res_s) == sorted(ex["S"]) and all(r["chosen"] == "W" and r["arm"] == "S" for r in res_s.values())
          and refused(B.e2_test_read, "S", verdict_path=vp, reference=READ_REF) is not None, res_s)
    e = refused(B.e2_test_read, "W", verdict_path=vp, reference=READ_REF)
    check("a second call is refused, naming the recorded argv", e is not None and "prepared once" in str(e)
          and "sbatch" in str(e), e)
    vp.write_text(vp.read_text().replace('"decided"', '"decided" ', 1))
    e = refused(B.e2_test_read, "W", verdict_path=vp, reference=READ_REF)
    check("  and after the verdict file changed, says so", e is not None and "the verdict changed" in str(e), e)
    for name in ("attempt.json", "run.json", "scores/test.json"):
        exn, vpn = read_world("e2n", qualifying=("W",), chosen="W")
        p = C.INC_DIR / exn["W"][1] / "runs" / "e2test__s1" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
        e = refused(B.e2_test_read, "W", verdict_path=vpn, reference=READ_REF)
        check("%s in one experiment's e2test run: refused, no spec written in any of the arm's experiments" % name,
              e is not None and "read once" in str(e) and no_specs(exn, "W"), e)
    exw, vpw = read_world("e2w", qualifying=("W",), chosen="W")
    (C.INC_DIR / exw["W"][2] / "runs" / "base__s2" / "weights" / "final.pt").write_bytes(os.urandom(256))
    e = refused(B.e2_test_read, "W", verdict_path=vpw, reference=READ_REF)
    check("base weights replaced since the verdict: refused, nothing written", e is not None
          and "the verdict was decided on" in str(e) and no_specs(exw, "W"), e)

    print("e2-test-report")
    rep = B.e2_test_report("W", verdict_path=vp, out_dir=vp.parent, reference=READ_REF)
    check("pending while the test scores are missing", rep["status"] == "pending" and len(rep["missing"]) == 6, rep)
    tw = {0: 0.885, 1: 0.889, 2: 0.887}
    wsha = {}
    for e in ex["W"] + ex["S"]:
        s = int(e[-1])
        wsha[e] = json.loads((C.INC_DIR / e / B.E2_TEST_RECORD).read_text())["weights_sha256"]
        _test_score(e, s, tw[s] - (0.004 if e in ex["S"] else 0.0), wsha[e])
        q = C.INC_DIR / READ_REF / "runs" / ("final__base__s%d" % s) / "scores" / "test.json"
        rw = json.loads((C.INC_DIR / READ_REF / "runs" / ("final__base__s%d" % s) / "run.json").read_text())
        q.write_text(json.dumps(dict({"exam": "test", "production": True, "species_map50_95": 0.8786 + 0.001 * (s - 1),
                                      "agnostic_map50_95": 0.8901, "weights_sha256": rw["weights_sha256"]},
                                     **TEST_STAMPS)))
    rep = B.e2_test_report("W", verdict_path=vp, out_dir=vp.parent, reference=READ_REF)
    check("complete: 12-class %.4f +- %.4f against the reference's %.4f, D_test %.4f, gap to 0.90 %.4f; the chosen "
          "arm is the headline; the read records' and the scores' sha256s and the shared stamps recorded"
          % (rep["arm_scores"]["twelve"]["mean"], rep["arm_scores"]["twelve"]["sd"],
             rep["reference"]["scores"]["twelve"]["mean"], rep["d_test"]["twelve"], rep["gap_to_target"]["arm"]),
          rep["status"] == "complete" and abs(rep["arm_scores"]["twelve"]["mean"] - 0.887) < 1e-9
          and abs(rep["arm_scores"]["twelve"]["sd"] - 0.002) < 1e-9
          and abs(rep["reference"]["scores"]["twelve"]["mean"] - 0.8786) < 1e-9
          and abs(rep["gap_to_target"]["arm"] - 0.013) < 1e-9 and abs(rep["d_test"]["twelve"] - 0.0084) < 1e-9
          and rep["headline"] is True and rep["chosen"] == "W" and rep["stamps"] == TEST_STAMPS
          and [x["weights_sha256"] for x in rep["inputs"]] == [wsha[e] for e in ex["W"]]
          and all(x["read_record_sha256"] and x["test_sha256"] for x in rep["inputs"])
          and all(x["test_sha256"] for x in rep["reference"]["inputs"])
          and (vp.parent / "e2_test_W.json").is_file() and (vp.parent / "e2_test_W.md").is_file(), rep)
    rep_s = B.e2_test_report("S", verdict_path=vp, out_dir=vp.parent, reference=READ_REF)
    check("  the qualifying arm not chosen (S): reported, not the headline, the report saying whose is",
          rep_s["status"] == "complete" and rep_s["headline"] is False and rep_s["chosen"] == "W"
          and "headline test number is E2-W's" in (vp.parent / "e2_test_S.md").read_text(), rep_s.get("headline"))
    ties = {}
    p0 = C.INC_DIR / ex["W"][0] / "runs" / "e2test__s0" / "scores" / "test.json"
    with saved(p0):
        _test_score(ex["W"][0], 0, 0.95, "f" * 64)
        ties["a test score from other weights (a resubmitted spec)"] = (
            refused(B.e2_test_report, "W", verdict_path=vp, out_dir=vp.parent, reference=READ_REF), "not the verdict's model")
    with saved(p0):
        _test_score(ex["W"][0], 0, 0.885, wsha[ex["W"][0]], stamps=dict(TEST_STAMPS, manifest_sha256="u" * 64))
        ties["a test score on another test manifest"] = (
            refused(B.e2_test_report, "W", verdict_path=vp, out_dir=vp.parent, reference=READ_REF), "manifest_sha256")
    q0 = C.INC_DIR / READ_REF / "runs" / "final__base__s0" / "scores" / "test.json"
    with saved(q0):
        q0.write_text(json.dumps(dict(json.loads(q0.read_text()), scorer_sha256="d" * 64)))
        ties["a reference score from another scorer"] = (
            refused(B.e2_test_report, "W", verdict_path=vp, out_dir=vp.parent, reference=READ_REF), "scorer_sha256")
    with saved(q0):
        q0.write_text(json.dumps(dict(json.loads(q0.read_text()), weights_sha256="e" * 64)))
        ties["a reference score from weights its final run does not hold"] = (
            refused(B.e2_test_report, "W", verdict_path=vp, out_dir=vp.parent, reference=READ_REF), "its final run")
    r0 = C.INC_DIR / ex["W"][1] / B.E2_TEST_RECORD
    with saved(r0):
        r0.unlink()
        ties["no e2_test_read.json for one input"] = (
            refused(B.e2_test_report, "W", verdict_path=vp, out_dir=vp.parent, reference=READ_REF), "no test read prepared")
    with saved(r0):
        r0.write_text(json.dumps(dict(json.loads(r0.read_text()), weights_sha256="a" * 64)))
        ties["a read record on other weights than the verdict's input"] = (
            refused(B.e2_test_report, "W", verdict_path=vp, out_dir=vp.parent, reference=READ_REF), "the verdict read")
    bad = {k: str(e)[:200] for k, (e, frag) in ties.items() if e is None or frag not in str(e)}
    check("the report refuses scores that are not the prepared read's: %s" % "; ".join(ties), not bad, bad)
    p = C.INC_DIR / ex["W"][0] / "runs" / "e2test__s0" / "scores" / "test.json"
    p.write_text(json.dumps(dict(json.loads(p.read_text()), production=False)))
    e = refused(B.e2_test_report, "W", verdict_path=vp, out_dir=vp.parent, reference=READ_REF)
    check("a test-mode score is refused in production", e is not None and "test-mode" in str(e), e)
    check("e2-test-report refuses an arm that did not qualify",
          refused(B.e2_test_report, "S", verdict_path=vp1, out_dir=vp1.parent, reference=READ_REF) is not None)
    exo, vpo = read_world("e2r2", qualifying=("W",), chosen="W", testing_allowed=True)
    check("  and a verdict that admitted test-mode files, in production",
          "pre-registered parameters" in str(refused(B.e2_test_report, "W", verdict_path=vpo, out_dir=vpo.parent,
                                                     reference=READ_REF)))
    check("no test-read file is on the platform's evidence (capacity/e2_test_W.json, <exp>/e2_test_read.json, the "
          "report), while e2_v1.json and e2_rescore.json are",
          not E.allowed("capacity/e2_test_W.json") and not E.allowed("capacity/e2_test_W.md")
          and not E.allowed("e2_w_m640_seed0/e2_test_read.json") and not E.allowed("capacity/e2_v1_report.json")
          and E.allowed("capacity/e2_v1.json") and E.allowed("capacity/e2_rescore.json"))


# ------------------------------------------------------------------ 14
def test_cli():
    print("the CLI")
    import io
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc1 = B.main(["rescore-e2", "--exp", "x"])
        rc2 = B.main(["e2-test-read"])
        rc3 = B.main(["e2-verdict", "--exp", "x"])
    check("rescore-e2 and e2-verdict refuse --exp (pre-registered), e2-test-read needs --e2: exit 1",
          rc1 == 1 and rc2 == 1 and rc3 == 1 and "pre-registered" in err.getvalue() and "needs --e2" in err.getvalue(),
          (rc1, rc2, rc3, err.getvalue()[-400:]))
    exp = RC.e2_exp("W", 2)
    drop(exp)
    rc = B.main(["build", "--exp", exp, "--manifest", str(base_v2()), "--seeds", "2", "--arm", "m640", "--role",
                 "baseline", "--e2", "W", "--testing-settings", json.dumps(W.TESTING), "--no-init"])
    s = json.loads((C.INC_DIR / exp / B.BUILD_SUMMARY).read_text()) if (C.INC_DIR / exp / B.BUILD_SUMMARY).exists() \
        else {}
    check("build --e2 W through the CLI (testing): E2-W seed 2 from E1-B's base__s2",
          rc == 0 and (s.get("e2") or {}).get("init", {}).get("run_id") == "base__s2", (rc, s.get("e2")))
    drop(exp)


def main():
    t0 = time.time()
    try:
        W.build_world()
        setup_world()
        test_constants_and_recipes()
        test_build()
        test_train_production()
        test_whole_load()
        test_bootstrap()
        ex, out = test_rule()
        test_verdict_file(ex, out)
        test_rescore()
        test_test_read()
        test_cli()
    finally:
        shutil.rmtree(W.TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
