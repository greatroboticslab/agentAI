"""Protocol v3, Stage A: pilot_v4, the LR re-warm recipes on known truth
(docs/CONTINUOUS_LOOP.md §5.1; docs/INCREMENTAL_PROTOCOL.md, "Protocol v3").

    python -m weed_optimizer_framework.tools.inc2.pilot4 build   --exp pilot_v4 --from pilot_v3 --recipes x1a,x1b
                                                                  [--testing | --testing-settings JSON] [--no-init]
    python -m weed_optimizer_framework.tools.inc2.pilot4 verdict --exp pilot_v4 [--no-write]

build makes a 'chain' experiment for the pinned driver on pilot_v3's bins,
sha-identical: P0 and the seven increments [I1, I2, Bswap, I3, Breal, I4, I5]
are pilot_v3's exp.json entries (names, clean flags, sessions, planted notes),
each manifest copied byte for byte into pilot_v4/manifests/ and checked to
hash to the manifest_sha256 pilot_v3 recorded (the rows still point at
pilot_v3's label files, e.g. the relabelled Bswap copies, which are not
copied). What changes:
  * recipes: the chains named by --recipes, from x1a and x1b of the
    Protocol v3 table (inc2.recipes, arm n640). R0 is pilot_v3's own full
    chain and is not re-run; freeze and lora are out (L-6);
  * no truth arm (truth false): the reference is pilot_v3's recorded truth
    verdicts, read from its ledger at build and pinned into exp.json;
  * replay mode full and the gate block exactly pilot_v3's (flips net);
  * the base recipe must be pilot_v3's (the v3 cold recipe of n640, which is
    the v1 cold recipe) or the build refuses; seeds and init are pilot_v3's;
  * final exams dev and imageweeds (v2's exams, test not read: P10);
  * exp.json carries inc2.recipes.stamp(n640) and a "stage_a" block: the
    pre-registered survival rule, the truth verdicts, R0's record (its chain
    verdicts from pilot_v3's ledger and its final dev from pilot_v3's report,
    judged by the same rule) and the sha256 of pilot_v3's exp.json, ledger
    and report.
Before anything is written, every manifest passes inc2.train's check_manifest
and the splits v2 never-train guard (fail closed), and the definition passes
the pinned driver's validate_definition and check_definition_data. build
then sets INC_JOB_SCRIPT to run_inc2_job.sh and calls driver init (as
inc2.baseline does).

The survival rule (pre-registered, §5.1). An arm survives only if it
  * ACCEPTs I2, I3 and I5;
  * REJECTs Bswap and Breal (a HOLD is not a rejection);
  * agrees with pilot_v3's truth arm on >= 5 of the 7 steps
    (ACCEPT ~ helps, HOLD ~ neutral, REJECT ~ hurts);
  * ends with final dev >= 0.8084 - 2 x 0.0043 = 0.7998 (pilot_v3's T_final
    dev mean and sd; the chain's final incumbent's dev mAP50-95).
R0 passes on its record (final dev 0.8006, agreement 5/7).

verdict writes INC_DIR/<exp>/stage_a.json: per arm its step verdicts,
agreement, final dev and every check; status READY once the experiment is
done and every arm is complete (PENDING before, with what is missing);
survivors; the best survivor (most agreement, then the higher final dev,
then fewer epochs, i.e. the cheaper recipe); and segment1_recipes = r0 plus
the best survivor, or r0 alone (§5.1 Stage B). A READY record is what D30
waits for: from then on D30 reads the stream's own segments only.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path

from ..inc import common as C
from ..inc import driver as D
from ..inc import gate as G
from ..inc import pilot as P
from ..inc.scorer import TEST_ENV
from . import baseline as B
from . import recipes as RC
from . import train as T

BUILDER = "inc2.pilot4 build"
SOURCE_DEFAULT = "pilot_v3"
ALLOWED_RECIPES = ("x1a", "x1b")
FINAL_EXAMS = ("dev", "imageweeds")
STAGE_A_NAME = "stage_a.json"
STAGE_A_FORMAT = "inc2-stage-a/1"
READY, PENDING = "READY", "PENDING"
SEQUENCE = tuple(P.SEQUENCE)
R0_CHAIN = "full"                                  # pilot_v3's R0 chain
RULE = {"accepts": ["I2", "I3", "I5"], "rejects": ["Bswap", "Breal"], "min_agreement": 5,
        "steps": len(SEQUENCE), "tfinal_dev": {"mean": 0.8084, "sd": 0.0043}, "sd_mult": 2,
        "final_dev_min": 0.7998,
        "agreement_map": {G.ACCEPT: G.HELPS, G.HOLD: G.NEUTRAL, G.REJECT: G.HURTS},
        "source": "docs/CONTINUOUS_LOOP.md 5.1 (pre-registered)"}
TFINAL_TOLERANCE = 5e-5            # pilot_v3's report must reproduce the rounded T_final numbers


class Pilot4Error(RuntimeError):
    """A condition under which Stage A must not be built or judged."""


def log(msg):
    print("[inc2.pilot4] %s" % msg, flush=True)


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _sha(path):
    try:
        return C.sha256_file(path)
    except OSError:
        return None


def read_ledger(path):
    out = []
    try:
        with open(path) as fh:
            for ln in fh:
                if ln.strip():
                    out.append(json.loads(ln))
    except OSError as e:
        raise Pilot4Error("cannot read %s (%s)" % (path, e))
    except ValueError as e:
        raise Pilot4Error("%s holds a line that is not JSON (%s)" % (path, e))
    return out


# ------------------------------------------------------------------- rule
def survival(verdicts, truth, final_dev, rule=RULE):
    """The pre-registered checks for one arm: verdicts {step: ACCEPT|HOLD|
    REJECT}, truth {step: helps|neutral|hurts}, final_dev (None = unknown)."""
    steps = list(SEQUENCE)
    missing = [s for s in steps if s not in verdicts]
    agree = [s for s in steps if s in verdicts and rule["agreement_map"].get(verdicts[s]) == truth.get(s)]
    checks = {
        "accepts": {s: verdicts.get(s) == G.ACCEPT for s in rule["accepts"]},
        "rejects": {s: verdicts.get(s) == G.REJECT for s in rule["rejects"]},
        "agreement": {"agree": len(agree), "of": len(steps), "min": rule["min_agreement"], "steps": agree,
                      "passed": len(agree) >= rule["min_agreement"]},
        "final_dev": {"value": final_dev, "min": rule["final_dev_min"],
                      "passed": final_dev is not None and final_dev >= rule["final_dev_min"]},
    }
    complete = not missing and final_dev is not None
    survives = (complete and all(checks["accepts"].values()) and all(checks["rejects"].values())
                and checks["agreement"]["passed"] and checks["final_dev"]["passed"])
    return {"verdicts": {s: verdicts.get(s) for s in steps}, "checks": checks, "agreement": len(agree),
            "final_dev": final_dev, "complete": complete, "missing": missing + ([] if final_dev is not None
                                                                                else ["final dev"]),
            "survives": bool(survives)}


def chain_verdicts(entries, chain):
    """{step name: verdict} of a chain's gate entries in a ledger."""
    return {e["step"]: e["decision"]["verdict"] for e in entries
            if e.get("type") == "gate" and e.get("chain") == chain}


def truth_verdicts(entries):
    return {e["step"]: e["detail"]["verdict"] for e in entries if e.get("type") == "truth"}


def report_final_dev(report, model_prefix):
    """The dev 12-class mean of a report.json 'final' row whose model starts
    with model_prefix, or None."""
    for f in (report or {}).get("final") or []:
        if str(f.get("model", "")).startswith(model_prefix):
            return (((f.get("exams") or {}).get("dev") or {}).get("twelve") or {}).get("mean")
    return None


def reference(source):
    """What Stage A reads of the source pilot: its definition, truth verdicts,
    R0's record and the T_final cross-check, with the sha256 of each file."""
    root = C.INC_DIR / source
    defn = _read_json(root / "exp.json")
    if not isinstance(defn, dict):
        raise Pilot4Error("%s has no readable exp.json" % root)
    entries = read_ledger(root / "ledger.jsonl")
    truth = truth_verdicts(entries)
    missing = [s for s in SEQUENCE if s not in truth]
    if missing:
        raise Pilot4Error("%s's ledger has no truth verdict for %s: Stage A's reference is the finished pilot's "
                          "truth arm" % (source, missing))
    rep_path = root / "report.json"
    rep = _read_json(rep_path)
    r0_dev = report_final_dev(rep, "chain %s" % R0_CHAIN)
    tfinal = None
    for f in (rep or {}).get("final") or []:
        if str(f.get("model", "")).startswith("T_final"):
            tfinal = ((f.get("exams") or {}).get("dev") or {}).get("twelve")
    cross = None
    if tfinal and tfinal.get("mean") is not None and tfinal.get("sd") is not None:
        cross = {"mean": tfinal["mean"], "sd": tfinal["sd"],
                 "matches_rule": (abs(tfinal["mean"] - RULE["tfinal_dev"]["mean"]) <= TFINAL_TOLERANCE
                                  and abs(tfinal["sd"] - RULE["tfinal_dev"]["sd"]) <= TFINAL_TOLERANCE)}
        if not cross["matches_rule"]:
            raise Pilot4Error("%s's T_final dev is %.5f +- %.5f; the pre-registered rule reads %.4f +- %.4f"
                              % (source, tfinal["mean"], tfinal["sd"], RULE["tfinal_dev"]["mean"],
                                 RULE["tfinal_dev"]["sd"]))
    r0 = survival(chain_verdicts(entries, R0_CHAIN), truth, r0_dev)
    return {"exp": source, "definition": defn, "truth": {s: truth[s] for s in SEQUENCE},
            "r0": dict(r0, chain=R0_CHAIN, final_dev_source=str(rep_path) if r0_dev is not None else None),
            "tfinal_crosscheck": cross,
            "files": {"exp_json": {"path": str(root / "exp.json"), "sha256": _sha(root / "exp.json")},
                      "ledger": {"path": str(root / "ledger.jsonl"), "sha256": _sha(root / "ledger.jsonl")},
                      "report": {"path": str(rep_path), "sha256": _sha(rep_path)}}}


# ------------------------------------------------------------------- build
def check_source(defn):
    if defn.get("type") != "chain":
        raise Pilot4Error("the source is a %r experiment, not a chain" % defn.get("type"))
    names = [s.get("name") for s in defn.get("steps") or []]
    if names != list(SEQUENCE):
        raise Pilot4Error("the source's steps are %s, not the pilot's sequence %s" % (names, list(SEQUENCE)))
    if defn.get("replay_mode") != "full":
        raise Pilot4Error("the source's replay mode is %r; Stage A compares with pilot_v3's full rehearsal"
                          % defn.get("replay_mode"))
    if not isinstance(defn.get("gate"), dict):
        raise Pilot4Error("the source has no gate block")
    if defn["base"].get("recipe") != RC.cold(RC.DEFAULT_ARM):
        raise Pilot4Error("the source's base recipe is not the Protocol v3 cold recipe of %s: P0 would be trained "
                          "differently" % RC.DEFAULT_ARM)


def build_definition(exp="pilot_v4", source=SOURCE_DEFAULT, recipes=ALLOWED_RECIPES, testing=False):
    """Check every input, copy the bins and write the summary; returns
    (definition, summary)."""
    recipes = [r.strip() for r in (recipes.split(",") if isinstance(recipes, str) else recipes) if r.strip()]
    bad = [r for r in recipes if r not in ALLOWED_RECIPES]
    if not recipes or bad or len(set(recipes)) != len(recipes):
        why = ("r0 is %s's own %s chain and is not re-run" % (source, R0_CHAIN) if "r0" in bad else
               "; ".join(RC.EXCLUDED_TRAINERS.get(r, "not a Stage A recipe") for r in bad) or "none given")
        raise Pilot4Error("--recipes %s: Stage A runs %s (%s)" % (recipes, list(ALLOWED_RECIPES), why))
    try:
        testing = P._check_testing(testing)
    except P.PilotError as e:
        raise Pilot4Error(str(e))
    ref = reference(source)
    src = ref["definition"]
    check_source(src)
    arm_rec = RC.resolve_arm(RC.DEFAULT_ARM, repo=C.REPO, require_weights=not testing)
    paths = D.Paths(exp)
    try:
        P._check_new(paths)
    except P.PilotError as e:
        raise Pilot4Error(str(e))
    lock = B.v2_lock_status(testing=bool(testing))
    production = not testing
    entries = [("base", src["base"])] + [("step", s) for s in src["steps"]]
    checked = {}
    for _kind, e in entries:
        p = Path(e["manifest"])
        if _sha(p) != e["manifest_sha256"]:
            raise Pilot4Error("%s's manifest %s does not hash to its recorded %s" % (source, p, e["manifest_sha256"][:12]))
        try:
            rows, _dh, info = B.check_training_manifest(p, production=production, what="%s bin %s" % (source, e["name"]))
        except B.BaselineError as err:
            raise Pilot4Error(str(err))
        checked[e["name"]] = {"images": len(rows), "guard": info["guard"], "boxes": info["train_class_counts"]}
    copies = {}
    for _kind, e in entries:
        src_path = Path(e["manifest"])
        dst = paths.manifests / src_path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_path, dst)
        if C.sha256_file(dst) != e["manifest_sha256"]:
            raise Pilot4Error("the copy of %s does not hash to %s" % (src_path, e["manifest_sha256"][:12]))
        copies[e["name"]] = str(dst)
    base = dict(copy.deepcopy(src["base"]), manifest=copies[src["base"]["name"]], source_manifest=src["base"]["manifest"])
    steps = [dict(copy.deepcopy(s), manifest=copies[s["name"]], source_manifest=s["manifest"]) for s in src["steps"]]
    rec = {r: RC.incremental(r, arm_rec["id"]) for r in recipes}
    warmup = P.warmup_table(rec, base["recipe"], base["n_images"],
                            [(s["name"], s["n_images"], s["clean"]) for s in steps], "full")
    stage_a = {"rule": RULE, "truth": ref["truth"], "r0_record": ref["r0"], "tfinal_crosscheck": ref["tfinal_crosscheck"],
               "source": {"exp": source, "files": ref["files"]}, "recipes": recipes,
               "segment1_rule": "Stage B runs r0 and the best survivor (most agreement, then higher final dev, then "
                                "fewer epochs), or r0 alone"}
    defn = {"exp": exp, "type": "chain", "builder": BUILDER, "testing": testing, "replay_mode": "full",
            "gate": copy.deepcopy(src["gate"]), "seeds": list(src["seeds"]), "decision_exam": D.DECISION_EXAM,
            "final_exams": list(FINAL_EXAMS), "base": base, "steps": steps, "recipes": rec, "truth": False,
            "effective_warmup": warmup, "attribution_scope": copy.deepcopy(src.get("attribution_scope")),
            "stage_a": stage_a,
            "splits_v2": {"lock": lock["lock"], "lock_sha256": lock["lock_sha256"],
                          "nevertrain_sha256": lock["index_sha256"]}}
    defn.update(RC.stamp(arm_rec))
    if src.get("init_weights") not in (None, arm_rec["model"]):
        raise Pilot4Error("the source's init weights %r are not %s" % (src.get("init_weights"), arm_rec["model"]))
    try:
        D.validate_definition(json.loads(json.dumps(defn)))
        D.check_definition_data(defn)
    except D.DriverError as e:
        raise Pilot4Error("the pinned driver refuses the definition: %s" % e)
    summary = {"exp": exp, "testing": bool(testing), "built_utc": D._utc(), "builder": BUILDER, "source": source,
               "recipes": rec, "cold_recipe": base["recipe"], "bins": checked,
               "sha_identical": {n: True for n in copies}, "stage_a": stage_a, "splits_v2": lock, "arm": arm_rec,
               "warmup": warmup}
    D._write_json(paths.root / P.BUILD_SUMMARY, summary)
    return defn, summary


def build(exp="pilot_v4", source=SOURCE_DEFAULT, recipes=ALLOWED_RECIPES, testing=False, backend=None, init=True,
          quiet=False):
    defn, summary = build_definition(exp, source, recipes, testing)
    r0 = defn["stage_a"]["r0_record"]
    log("%s from %s: chains %s on the same %d bins, no truth arm; R0 on its record: agreement %d/7, final dev %s, "
        "%s" % (exp, source, sorted(defn["recipes"]), len(defn["steps"]), r0["agreement"], r0["final_dev"],
                "survives" if r0["survives"] else "does not survive"))
    result = None
    if init:
        B.ensure_job_script(testing=bool(defn["testing"]))
        result = D.Driver(exp, backend=backend, quiet=quiet).init(defn)
    return summary, defn, result


# ----------------------------------------------------------------- verdict
def verdict(exp="pilot_v4", write=True):
    """Stage A's record (module docstring)."""
    root = C.INC_DIR / exp
    defn = _read_json(root / "exp.json")
    if not isinstance(defn, dict) or not isinstance(defn.get("stage_a"), dict):
        raise Pilot4Error("%s is not a Stage A experiment (no stage_a block)" % root)
    sa = defn["stage_a"]
    state = _read_json(root / "state.json") or {}
    entries = read_ledger(root / "ledger.jsonl") if (root / "ledger.jsonl").is_file() else []
    arms, inputs = {}, {}
    for r in sorted(defn["recipes"]):
        p = root / "runs" / ("final__%s__incumbent" % r) / "scores" / "dev.json"
        d = _read_json(p)
        fd = None
        if isinstance(d, dict) and d.get("exam") == "dev" and (d.get("production") is True or defn.get("testing")):
            fd = float(d["map50_95"])
        inputs[r] = {"final_dev": {"path": str(p), "sha256": _sha(p)}}
        res = survival(chain_verdicts(entries, r), sa["truth"], fd, sa["rule"])
        res["recipe"] = RC.incremental(r, (defn.get("arm") or {}).get("id") or RC.DEFAULT_ARM)
        arms[r] = res
    ready = bool(state.get("done")) and all(a["complete"] for a in arms.values())
    survivors = sorted(r for r, a in arms.items() if a["survives"])
    best = None
    if survivors:
        best = max(survivors, key=lambda r: (arms[r]["agreement"], arms[r]["final_dev"],
                                             -arms[r]["recipe"]["epochs"], -ALLOWED_RECIPES.index(r)))
    doc = {"format": STAGE_A_FORMAT, "exp": exp, "status": READY if ready else PENDING,
           "generated_utc": D._utc(), "rule": sa["rule"], "truth": sa["truth"], "r0": sa["r0_record"],
           "arms": arms, "survivors": survivors, "best_survivor": best,
           "segment1_recipes": ["r0"] + ([best] if best else []),
           "pending": {} if ready else {"experiment_done": bool(state.get("done")),
                                        "incomplete_arms": {r: a["missing"] for r, a in arms.items()
                                                            if not a["complete"]}},
           "inputs": {"exp_json": {"path": str(root / "exp.json"), "sha256": _sha(root / "exp.json")},
                      "ledger": {"path": str(root / "ledger.jsonl"), "sha256": _sha(root / "ledger.jsonl")},
                      "state_generation": state.get("generation"), "finals": inputs,
                      "source": sa.get("source")},
           "d30": "Stage A READY: D30 reads the stream's own segments only, whatever survived (5.1)" if ready
                  else "Stage A not READY: D30 holds the TRAIN lane (5.1)"}
    if write:
        doc["out"] = str(root / STAGE_A_NAME)
        B._write_json(doc["out"], doc)
    log("%s: %s; survivors %s; segment 1 recipes %s" % (exp, doc["status"], survivors, doc["segment1_recipes"]))
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description="Protocol v3 Stage A (pilot_v4).")
    ap.add_argument("command", choices=("build", "verdict"))
    ap.add_argument("--exp", default="pilot_v4")
    ap.add_argument("--from", dest="source", default=SOURCE_DEFAULT)
    ap.add_argument("--recipes", default=",".join(ALLOWED_RECIPES))
    ap.add_argument("--testing", action="store_true", help="needs %s=1" % TEST_ENV)
    ap.add_argument("--testing-settings", default=None)
    ap.add_argument("--no-init", action="store_true")
    ap.add_argument("--no-write", action="store_true", help="verdict: print only")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        if a.command == "build":
            testing = P.testing_arg(a.testing, a.testing_settings)
            build(a.exp, a.source, a.recipes, testing=testing, init=not a.no_init, quiet=a.quiet)
        else:
            doc = verdict(a.exp, write=not a.no_write)
            print(json.dumps({k: doc[k] for k in ("status", "survivors", "best_survivor", "segment1_recipes")}))
    except (Pilot4Error, B.BaselineError, P.PilotError, D.DriverError, RC.RecipeError, T.RunError) as e:
        print("[inc2.pilot4] ERROR: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
