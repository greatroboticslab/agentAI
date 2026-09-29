#!/usr/bin/env python3
"""Protocol v3 Stage A (inc2/pilot4.py): pilot_v4 on pilot_v3's bins
(docs/CONTINUOUS_LOOP.md §5.1, §9 group B).

A synthetic 'pilot_v3' (exp.json with P0 and the seven bins, a ledger with its
truth verdicts and its full chain's gate verdicts as recorded on the cluster,
a report.json with the full chain's final dev and T_final) in the world of
tests/test_inc2_train.py.

What is pinned:
- build: the bins are sha-identical to the source's exp.json entries (same
  bytes, names, clean flags, sessions), the chains are x1a and x1b from the v3
  table, there is no truth arm, the gate block and replay mode are the
  source's, final exams are dev and imageweeds, the definition passes the
  pinned driver's checks, and stage_a pins the rule, the source's truth
  verdicts, R0's record (it survives: 5/7, final dev 0.8006) and the sha256
  of the source's files;
- refusals: r0 (pilot_v3's own chain), freeze, lora, an unknown or repeated
  recipe; a source missing a truth verdict, with another base recipe, a bin
  that no longer hashes as recorded, a T_final that is not the rule's, a
  source that is not a chain, a bin holding an hflip copy of a test image
  (the v2 guard over every bin); a second build;
- init (FakeBackend): the first submission is the three cold base runs, each
  spec one the v2 executor accepts, with INC_JOB_SCRIPT = run_inc2_job.sh;
- verdict: READY once done with every arm complete, survivors, the best
  survivor (agreement, then final dev, then fewer epochs), segment1_recipes;
  a HOLD on a planted bin is not a rejection (the arm fails with 5/7);
  PENDING before, naming what is missing; the CLI;
- the rule on pilot_v3's recorded chains (local artifacts): R0 survives 5/7
  with final dev 0.8006; freeze and lora agree on 3/7 and fail (L-6).

Run:  python3 tests/test_inc2_pilot4.py
"""
import contextlib
import copy
import io
import json
import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_train as W  # noqa: E402  (sets INC_DIR / REPO to its own temporary dirs first)

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import driver as D  # noqa: E402
from weed_optimizer_framework.tools.inc import pilot as P  # noqa: E402
from weed_optimizer_framework.tools.inc2 import baseline as B  # noqa: E402
from weed_optimizer_framework.tools.inc2 import pilot4 as P4  # noqa: E402
from weed_optimizer_framework.tools.inc2 import recipes as RC  # noqa: E402
from weed_optimizer_framework.tools.inc2 import train as T  # noqa: E402

FAILURES = W.FAILURES
check = W.check
REAL_INC = W.ROOT / "results" / "framework" / "inc"
TRUTH = {"I1": "hurts", "I2": "helps", "Bswap": "hurts", "I3": "helps", "Breal": "hurts", "I4": "neutral",
         "I5": "helps"}
FULL = {"I1": "ACCEPT", "I2": "ACCEPT", "Bswap": "REJECT", "I3": "ACCEPT", "Breal": "REJECT", "I4": "ACCEPT",
        "I5": "ACCEPT"}


def make_source(name, Wd, extra_rows, drop_truth=None, base_recipe=None, tfinal=(0.8083685911566377, 0.004286248),
                typ="chain"):
    root = C.INC_DIR / name
    mdir = root / "manifests"
    base_rows = Wd["train_core"]
    entries = []
    for i, step in enumerate(P.SEQUENCE):
        rows = extra_rows[3 * i:3 * i + 3]
        path = mdir / ("%s.jsonl" % step)
        sha = C.write_manifest(path, rows)
        e = {"name": step, "manifest": str(path), "manifest_sha256": sha, "n_images": len(rows),
             "clean": step in P.CLEAN}
        e.update({"sessions": ["s%d" % i]} if step in P.CLEAN else {"planted": "planted %s" % step})
        entries.append(e)
    bpath = mdir / "P0.jsonl"
    bsha = C.write_manifest(bpath, base_rows)
    cold = base_recipe or P.cold_recipe()
    defn = {"exp": name, "type": typ, "builder": "inc.pilot build", "testing": False, "replay_mode": "full",
            "gate": {"flips_mode": "net"}, "seeds": [0, 1, 2], "init_weights": "yolo11n.pt", "decision_exam": "dev",
            "final_exams": ["dev", "ood22", "ood23", "imageweeds", "test"],
            "base": {"name": "P0", "manifest": str(bpath), "manifest_sha256": bsha, "n_images": len(base_rows),
                     "recipe": cold, "sessions": ["a"]},
            "steps": entries, "recipes": P.inc_recipes(), "truth": True, "truth_recipe": cold,
            "attribution_scope": P.ATTRIBUTION_SCOPE}
    (root / "exp.json").write_text(json.dumps(defn))
    led = []
    for k, step in enumerate(P.SEQUENCE, 1):
        if step != drop_truth:
            led.append({"id": "truth/%d" % k, "type": "truth", "k": k, "step": step,
                        "detail": {"verdict": TRUTH[step], "p": 0.5}})
        led.append({"id": "gate/full/%d" % k, "type": "gate", "chain": "full", "k": k, "step": step,
                    "decision": {"verdict": FULL[step]}})
    (root / "ledger.jsonl").write_text("".join(json.dumps(x) + "\n" for x in led))
    rep = {"final": [{"model": "chain full: final incumbent", "exams": {"dev": {"twelve": {"mean": 0.8005601516, "n": 1}}}},
                     {"model": "T_final (union of clean data)", "exams": {"dev": {"twelve": {"mean": tfinal[0],
                                                                                             "sd": tfinal[1], "n": 3}}}}]}
    (root / "report.json").write_text(json.dumps(rep))
    return root


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except (P4.Pilot4Error, B.BaselineError, D.DriverError, RC.RecipeError) as e:
        return e
    return None


def test_build(Wd, extra):
    print("build")
    src = make_source("pilot_v3", Wd, extra)
    defn, summ = P4.build_definition("pilot_v4", "pilot_v3", "x1a,x1b", testing=W.TESTING)
    sdef = json.loads((src / "exp.json").read_text())
    same = all(s["manifest_sha256"] == t["manifest_sha256"] and W.sha(s["manifest"]) == W.sha(t["manifest"])
               and s["name"] == t["name"] and s["clean"] == t["clean"] and s["source_manifest"] == t["manifest"]
               and s.get("sessions") == t.get("sessions") and s.get("planted") == t.get("planted")
               and s["manifest"] != t["manifest"]
               for s, t in zip(defn["steps"], sdef["steps"]))
    check("the seven bins and P0 are pilot_v3's, sha-identical, copied into pilot_v4/manifests",
          same and len(defn["steps"]) == 7 and defn["base"]["manifest_sha256"] == sdef["base"]["manifest_sha256"]
          and W.sha(defn["base"]["manifest"]) == sdef["base"]["manifest_sha256"]
          and pathlib.Path(defn["base"]["manifest"]).parent == C.INC_DIR / "pilot_v4" / "manifests")
    check("chains x1a and x1b from the v3 table, no truth arm, the source's gate and replay mode, finals dev and "
          "imageweeds, the source's base recipe and seeds",
          defn["recipes"] == {"x1a": RC.incremental("x1a"), "x1b": RC.incremental("x1b")} and defn["truth"] is False
          and "truth_recipe" not in defn and defn["gate"] == sdef["gate"] and defn["replay_mode"] == "full"
          and defn["final_exams"] == ["dev", "imageweeds"] and defn["base"]["recipe"] == sdef["base"]["recipe"]
          and defn["seeds"] == sdef["seeds"] and defn["protocol"] == "v3" and defn["arm"]["id"] == "n640")
    D.validate_definition(json.loads(json.dumps(defn)))
    D.check_definition_data(defn)
    check("the pinned driver's validate_definition and check_definition_data accept it", True)
    sa = defn["stage_a"]
    check("stage_a pins the rule, the truth verdicts, R0's record (survives 5/7, final dev 0.8006) and the "
          "source files' sha256", sa["truth"] == TRUTH and sa["r0_record"]["survives"]
          and sa["r0_record"]["agreement"] == 5 and abs(sa["r0_record"]["final_dev"] - 0.80056) < 1e-4
          and sa["rule"]["final_dev_min"] == 0.7998 and sa["tfinal_crosscheck"]["matches_rule"]
          and sa["source"]["files"]["ledger"]["sha256"] == W.sha(src / "ledger.jsonl"), sa["r0_record"])
    check("the build summary records every bin's guard (nothing refused)",
          all(v["guard"]["refused"] == 0 for v in summ["bins"].values()) and len(summ["bins"]) == 8)

    for what, recipes in (("r0 (pilot_v3's own chain)", "r0"), ("freeze (L-6)", "freeze"), ("lora (L-6)", "lora"),
                          ("an unknown recipe", "x1c"), ("a repeated recipe", "x1a,x1a"), ("no recipe", "")):
        e = refused(P4.build_definition, "pilot_v4r", "pilot_v3", recipes, testing=W.TESTING)
        check("refused: %s" % what, e is not None, e)
    make_source("src_notruth", Wd, extra, drop_truth="Breal")
    make_source("src_recipe", Wd, extra, base_recipe=dict(P.cold_recipe(), epochs=90))
    make_source("src_tfinal", Wd, extra, tfinal=(0.8150, 0.0043))
    make_source("src_type", Wd, extra, typ="baseline")
    plant = W.planted("te_hflip_p4", Wd["test"][2]["image"], "hflip")
    leaky = list(extra)
    leaky[3 * P.SEQUENCE.index("I3")] = W.row_for("pl_p4", plant, [(0, .5, .5, .2, .2)], "cottonweeddet12/train")
    make_source("src_leak", Wd, leaky)
    tam = make_source("src_tamper", Wd, extra)
    (tam / "manifests" / "I3.jsonl").write_text((tam / "manifests" / "I3.jsonl").read_text() + "\n")
    for what, srcname, needle in (("a source missing a truth verdict", "src_notruth", "no truth verdict"),
                                  ("a source whose base recipe differs", "src_recipe", "base recipe"),
                                  ("a T_final that is not the rule's", "src_tfinal", "pre-registered"),
                                  ("a source that is not a chain", "src_type", "not a chain"),
                                  ("a bin holding an hflip copy of a test image", "src_leak", "never-train guard"),
                                  ("a bin that no longer hashes as recorded", "src_tamper", "does not hash")):
        e = refused(P4.build_definition, "pilot_v4s", srcname, "x1a", testing=W.TESTING)
        check("refused: %s" % what, e is not None and needle in str(e), e)
    check("... and none of them wrote an exp.json", not (C.INC_DIR / "pilot_v4s" / "exp.json").exists())


def test_init():
    print("init through the v2 job script")
    subs = []

    class Rec(D.FakeBackend):
        def submit(self, list_file, n, exp, job_name, log_dir, env=None, time_limit=None):
            subs.append((env or {}).get("INC_JOB_SCRIPT"))
            return super().submit(list_file, n, exp, job_name, log_dir, env=env, time_limit=time_limit)

    backend = Rec()
    shutil.rmtree(C.INC_DIR / "pilot_v4", ignore_errors=True)
    summ, defn, res = P4.build("pilot_v4", "pilot_v3", "x1a,x1b", testing=W.TESTING, backend=backend, quiet=True)
    specs = [json.loads(pathlib.Path(s).read_text()) for s in backend.submissions[0]["specs"]]
    ok = []
    for s in specs:
        T.validate_spec(s, pathlib.Path(s["out_dir"]) / "spec.json")
        arm, _ = T.experiment_arm(s["exp"])
        ok.append(not RC.deviations(s["kind"], s["recipe"], arm) and not T.init_check(s["kind"], s["init"], arm))
    check("the first submission is the three cold base runs with the v3 cold recipe, each spec accepted by "
          "the v2 executor, under INC_JOB_SCRIPT = run_inc2_job.sh",
          sorted(s["run_id"] for s in specs) == ["base__s0", "base__s1", "base__s2"] and all(ok)
          and subs == [str(B.job_script_path())] and backend.submissions[0]["time_limit"] == D.COLD_TIME_LIMIT,
          (subs, [s["run_id"] for s in specs]))
    e = refused(P4.build_definition, "pilot_v4", "pilot_v3", "x1a", testing=W.TESTING)
    check("a built pilot_v4 is not built again", e is not None and "already built" in str(e), e)
    return defn


def write_arm(root, r, verdicts, final_dev):
    f = root / "runs" / ("final__%s__incumbent" % r) / "scores" / "dev.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    if final_dev is not None:
        f.write_text(json.dumps({"exam": "dev", "map50_95": final_dev, "production": False}))
    return [{"id": "gate/%s/%d" % (r, k), "type": "gate", "chain": r, "k": k, "step": step,
             "decision": {"verdict": verdicts[step]}} for k, step in enumerate(P.SEQUENCE, 1)]


def test_verdict(defn):
    print("verdict")

    def setup(name, arms, done=True):
        root = C.INC_DIR / name
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        d = copy.deepcopy(defn)
        d["exp"] = name
        (root / "exp.json").write_text(json.dumps(d))
        (root / "state.json").write_text(json.dumps({"done": done, "generation": 9}))
        led = []
        for r, (v, fd) in arms.items():
            led += write_arm(root, r, v, fd)
        (root / "ledger.jsonl").write_text("".join(json.dumps(x) + "\n" for x in led))
        return root

    x1a = dict(FULL, I4="HOLD")            # 6/7
    x1b = dict(FULL, Bswap="ACCEPT")       # accepts a planted bin
    setup("p4_a", {"x1a": (x1a, 0.8050), "x1b": (x1b, 0.8100)})
    doc = P4.verdict("p4_a")
    a, b = doc["arms"]["x1a"], doc["arms"]["x1b"]
    check("READY: x1a survives (6/7, final dev 0.805 >= 0.7998); x1b does not (it ACCEPTs Bswap); segment 1 "
          "runs r0 and x1a", doc["status"] == "READY" and a["survives"] and a["agreement"] == 6
          and not b["survives"] and b["checks"]["rejects"]["Bswap"] is False and doc["survivors"] == ["x1a"]
          and doc["best_survivor"] == "x1a" and doc["segment1_recipes"] == ["r0", "x1a"]
          and (C.INC_DIR / "p4_a" / "stage_a.json").is_file() and doc["r0"]["survives"], doc["arms"])
    setup("p4_b", {"x1a": (x1a, 0.8010), "x1b": (x1a, 0.8030)})
    setup("p4_c", {"x1a": (x1a, 0.8030), "x1b": (x1a, 0.8030)})
    setup("p4_d", {"x1a": (FULL, 0.7990), "x1b": (x1b, 0.8100)})
    db, dc, dd = P4.verdict("p4_b"), P4.verdict("p4_c"), P4.verdict("p4_d")
    check("the best survivor: equal agreement -> the higher final dev (x1b); equal dev too -> fewer epochs (x1a)",
          db["best_survivor"] == "x1b" and dc["best_survivor"] == "x1a", (db["survivors"], dc["survivors"]))
    check("no survivor (x1a's final dev 0.7990 < 0.7998, x1b accepts Bswap): segment 1 runs r0 alone",
          dd["status"] == "READY" and dd["survivors"] == [] and dd["segment1_recipes"] == ["r0"]
          and dd["arms"]["x1a"]["checks"]["final_dev"]["passed"] is False)
    setup("p4_h", {"x1a": (dict(x1a, Bswap="HOLD"), 0.8050), "x1b": (x1a, 0.8030)})
    dh = P4.verdict("p4_h")
    check("a HOLD on Bswap is not a rejection: that arm (accepts and agreement otherwise met, 5/7) does not "
          "survive; the other does", dh["arms"]["x1a"]["agreement"] == 5 and dh["arms"]["x1a"]["checks"]["agreement"]
          ["passed"] and dh["arms"]["x1a"]["checks"]["rejects"]["Bswap"] is False and not dh["arms"]["x1a"]["survives"]
          and dh["survivors"] == ["x1b"], dh["arms"]["x1a"]["checks"])
    setup("p4_e", {"x1a": (x1a, 0.8050), "x1b": (x1b, None)}, done=False)
    de = P4.verdict("p4_e")
    check("PENDING while the experiment is not done, naming the arm without a final dev",
          de["status"] == "PENDING" and de["pending"]["incomplete_arms"].get("x1b") == ["final dev"]
          and not de["pending"]["experiment_done"])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = P4.main(["verdict", "--exp", "p4_a", "--no-write"])
    check("the CLI prints the verdict", rc == 0 and json.loads(buf.getvalue().strip().splitlines()[-1])["status"]
          == "READY")


def test_recorded_pilot_v3():
    print("the rule on pilot_v3's recorded chains")
    led, rep = REAL_INC / "pilot_v3" / "ledger.jsonl", REAL_INC / "pilot_v3" / "report.json"
    if not (led.is_file() and rep.is_file()):
        print("  NOTE %s is not here; skipped" % led.parent)
        return
    entries = P4.read_ledger(led)
    report = json.loads(rep.read_text())
    truth = P4.truth_verdicts(entries)
    out = {ch: P4.survival(P4.chain_verdicts(entries, ch), truth, P4.report_final_dev(report, "chain %s" % ch))
           for ch in ("full", "freeze", "lora")}
    check("R0 (pilot_v3's full chain) survives on its record: 5/7, final dev 0.8006",
          out["full"]["survives"] and out["full"]["agreement"] == 5 and abs(out["full"]["final_dev"] - 0.8006) < 5e-5)
    check("freeze and lora agree on 3/7 and do not survive (L-6)",
          out["freeze"]["agreement"] == 3 and out["lora"]["agreement"] == 3
          and not out["freeze"]["survives"] and not out["lora"]["survives"])


def main():
    t0 = time.time()
    try:
        Wd = W.build_world()
        extra = W.make_rows("pv", 21, 700, "cottonweeddet12/train")
        test_build(Wd, extra)
        defn = test_init()
        test_verdict(defn)
        test_recorded_pilot_v3()
    finally:
        shutil.rmtree(W.TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
