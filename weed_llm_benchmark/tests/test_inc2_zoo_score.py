#!/usr/bin/env python3
"""The model-zoo audit's scoring passes (inc2/zoo.py score, exam-root): real
Ultralytics passes on the CPU in test mode (imgsz 64, batch 8, INC_SCORER_TESTING=1)
over the world of tests/test_inc2_zoo.py, with INC_DIR at the zoo's exam root
(docs/CONTINUOUS_LOOP.md, "Amendment (2026-10-04): Z1, the model-zoo audit
(pre-registered usage)").

What is pinned:
- a stage-A shard of b (converted), e (converted) and f (converted) on dev and
  test v1: records written once, stamped TEST-ZOO-<scorer sha256>, production
  false, the scorer's own stamp and production kept as locked_scorer_sha256 and
  scorer_production; a record's agnostic AP equals a direct S.score of the file
  under the zoo root (within 1e-9); test v1's per-source APs recompute the pass;
  e's species column is empty in its report row;
- a second run of the shard scores nothing (S.score is not called);
- a refusal (the unconverted source under the converted row's id: the scorer
  refuses its class space) is a final record, never retried; an exception is
  an error record retried once, then final; one message on three models
  stops the task (exit 1);
- task_deadline_s 0: every item not_scored_time, exit 0;
- the root checks: INC_DIR not a zoo root, a LOCK that is not the canonical
  one (exit 1); a file that does not hash as planned (an error record);
- exam-root with $LOCAL: the shard's exams copied as real files, a scored record
  says exam_root local and equals the Lustre root's score; too little room:
  the Lustre root;
- score --item ID:EXAM matches the shard's record;
- an INC row is read on dev alone: score --item on test or test v1, or a
  shard holding one of its items off dev, refuses before anything is scored;
- the pilot (a real subprocess, as the inventory runs it) inside a job that
  started 4.2 h ago scores every item (its deadline runs from its own start)
  and measures a rate for every exam with images; a pilot that measures
  nothing (task_deadline_s 0) refuses and is not marked done;
- nothing under the INC run directories changes; no score file appears there.

Run:  python3 tests/test_inc2_zoo_score.py
"""
import contextlib
import hashlib
import io
import json
import os
import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_zoo as T  # noqa: E402  (the zoo world; sets INC_DIR / REPO / INC_SCORER_TESTING first)

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc2 import zoo as Z  # noqa: E402

FAILURES = T.FAILURES
check = T.check
V = T.V
TMP = T.TMP


@contextlib.contextmanager
def as_root(root):
    """inc.common's paths at the zoo root for this process (what INC_DIR=<root>
    gives a scoring job's own process), INC_ZOO_SOURCE at the INC tree."""
    names = ("INC_DIR", "SPLITS_DIR", "EXAMS_DIR", "LOCK_PATH", "NEVER_TRAIN_INDEX")
    saved = {k: getattr(C, k) for k in names}
    src = str(saved["INC_DIR"])
    root = pathlib.Path(root)
    C.INC_DIR = root
    C.SPLITS_DIR = root / "splits" / "v1"
    C.EXAMS_DIR = root / "exams" / "v1"
    C.LOCK_PATH = C.SPLITS_DIR / "LOCK.json"
    C.NEVER_TRAIN_INDEX = C.SPLITS_DIR / "nevertrain_dhash.json"
    old = os.environ.get("INC_ZOO_SOURCE")
    os.environ["INC_ZOO_SOURCE"] = src
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(C, k, v)
        if old is None:
            os.environ.pop("INC_ZOO_SOURCE", None)
        else:
            os.environ["INC_ZOO_SOURCE"] = old


def item(key, exam):
    m = T.models_by_rel()[T.FIX[key]]
    got = Z.scorable(m, V)
    return {"model_id": m["model_id"], "exam": exam, "file": got[0], "file_sha256": got[1],
            "conversion": Z._conv_brief(got[2]), "predicted_s": 1.0}


def tree_hash(root):
    h = hashlib.sha256()
    for p in sorted(pathlib.Path(root).rglob("*")):
        if p.is_file():
            h.update(str(p).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def run(items, task="a_000", stage="a"):
    c, csha = T.conf()
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        return Z.run_items(V, c, csha, items, task, stage)


def pilot_checks():
    print("the pilot: its deadline runs from its own start; a pilot that measures no rate refuses")
    c, csha = T.conf()
    zd = Z.zoo_dir(V)
    e0 = None
    with T.env(INC_ZOO_TASK_DEADLINE_S="0"):
        try:
            T.quiet(Z.run_step, V, c, csha, "pilot", None)
        except Z.ZooRefused as e:
            e0 = e
    p0 = json.loads((zd / "pilot.json").read_text())
    exn = Z.read_exams(V)["exams"]
    with_images = sorted(e for e in Z.ALL_EXAMS if exn[e]["n_images"])
    check("a pilot that measures nothing (task_deadline_s 0: every item not_scored_time) refuses, says incomplete "
          "and is not marked done (a resubmission runs it again)",
          e0 is not None and "measured no rate" in str(e0) and p0["status"] == "incomplete"
          and sorted(p0["missing_exams"]) == with_images and p0["task_counts"] == {"not_scored_time": len(with_images)}
          and not Z.step_done(V, "pilot", csha), (e0, p0.get("status"), p0.get("task_counts")))
    e1, pr = None, {}
    with T.env(SLURM_JOB_START_TIME=str(int(time.time() - 4.2 * 3600))):
        try:
            pr, _o = T.quiet(Z.run_step, V, c, csha, "pilot", None)
        except Z.ZooRefused as e:
            e1 = e
    task = json.loads((zd / "tasks" / "pilot.json").read_text())
    check("inside an inventory job that started 4.2 h ago (past the 3.75 h task deadline) the pilot scores every item "
          "(its deadline runs from its own start): a measured rate for every exam with images; marked done",
          e1 is None and task["counts"] == {"scored": len(task["items"])} and len(task["items"]) == len(with_images)
          and pr.get("status") == "complete" and not pr.get("missing_exams")
          and all(pr["rates"].get(e) for e in with_images) and Z.step_done(V, "pilot", csha),
          (e1, task["counts"], pr.get("missing_exams")))
    # the pilot's records are real records: removed so the stage-A checks below start from none
    for it in Z.read_shard(V, "pilot")["items"]:
        p = Z.score_path(V, it["model_id"], it["exam"])
        if p.exists():
            p.unlink()


def main():
    T.world()
    T.run_steps("list", "meta", "provenance", "convert", "exams", "contamination")
    inc_before = tree_hash(C.INC_DIR / "zt_inc")
    root = Z.root_dir(V)
    pilot_checks()
    items = [item(k, e) for k in ("b_best", "e", "f") for e in ("dev", "test_v1")]
    print("a stage-A shard on dev and test v1 under the zoo root")
    with as_root(root):
        Z.check_root(V)
        rec = run(items)
    recs = {(i["model_id"], i["exam"]): Z.record_of(V, i["model_id"], i["exam"]) for i in items}
    ss = Z._sha_file(pathlib.Path(S.__file__).resolve())
    check("every item scored once: records stamped TEST-ZOO-<scorer>, production false, the scorer's own stamp kept",
          rec["counts"] == {"scored": 6} and all(k == "scored" and r["scorer_stamp"] == "TEST-ZOO-" + ss
                                                 and r["production"] is False
                                                 and r["result"]["locked_scorer_sha256"] == "TEST-" + ss
                                                 and r["result"]["scorer_production"] is False
                                                 for k, r in recs.values()), rec["counts"])
    rb = recs[(items[0]["model_id"], "dev")][1]
    with as_root(root):
        direct = S.score(items[0]["file"], "dev", TMP / "direct.json", imgsz=64, batch=8, device="cpu")
    check("a record's agnostic AP equals a direct S.score of the same file under the zoo root (1e-9)",
          abs(rb["result"]["agnostic_map50_95"] - direct["agnostic_map50_95"]) <= 1e-9,
          (rb["result"]["agnostic_map50_95"], direct["agnostic_map50_95"]))
    rt = recs[(items[1]["model_id"], "test_v1")][1]
    check("test v1: per-source APs from the same pass; the captured arrays reproduce its agnostic AP (1e-9); no "
          "12-class value kept", set(rt["per_source"]) == {"src0", "src1"} and rt["checks"]["recompute"] <= 1e-9
          and "map50_95" not in rt["result"] and "image_correct" not in rt["result"], rt.get("per_source"))
    c, csha = T.conf()
    rows = {r["rel"]: r for r in Z.build_rows(V, c)}
    check("(e) one class: its species column is empty in its report row, its agnostic dev is the record's",
          rows[T.FIX["e"]]["cols"]["dev_sp"] is None and rows[T.FIX["e"]]["cols"]["dev_ag"] ==
          recs[(items[2]["model_id"], "dev")][1]["result"]["agnostic_map50_95"])

    print("resumable")
    calls = []
    orig = S.score

    def counting(*a, **k):
        calls.append(a)
        return orig(*a, **k)
    S.score = counting
    try:
        with as_root(root):
            rec2 = run(items)
    finally:
        S.score = orig
    check("a second run of the shard scores nothing (S.score not called)", not calls and set(rec2["counts"]) ==
          {"kept_scored"}, rec2["counts"])

    print("refusals, errors, systemic stops")
    m_b = T.models_by_rel()[T.FIX["b_best"]]
    src = dict(items[0], exam="imageweeds", file=m_b["path"], file_sha256=m_b["model_id"])
    with as_root(root):
        r1 = run([src], task="t_ref")
        r1b = run([src], task="t_ref")
    kind, fr = Z.record_of(V, src["model_id"], "imageweeds")
    check("the unconverted source (its own class space) is refused by the scorer: a final refusal, not retried",
          r1["counts"] == {"refused": 1} and kind == "refused" and "INC class space" in fr["refusal"]
          and r1b["counts"] == {"kept_refused": 1}, (r1["counts"], r1b["counts"]))
    boom = {"n": 0}

    def raising(*a, **k):
        boom["n"] += 1
        raise ValueError("planted failure in the validator")
    it_e = dict(items[2], exam="imageweeds")
    S.score = raising
    try:
        with as_root(root):
            e1 = run([it_e], task="t_err")
            k1, er1 = Z.record_of(V, it_e["model_id"], "imageweeds")
            e2 = run([it_e], task="t_err")
            k2, er2 = Z.record_of(V, it_e["model_id"], "imageweeds")
            e3 = run([it_e], task="t_err")
    finally:
        S.score = orig
    check("an exception: an error record, retried once, then final (error_attempts 2); a final error is not retried",
          e1["counts"] == {"error": 1} and k1 is None and er1["attempts"] == 1 and e2["counts"] == {"error_final": 1}
          and k2 == "error_final" and e3["counts"] == {"kept_error_final": 1} and boom["n"] == 2,
          (e1["counts"], e2["counts"], e3["counts"], boom))
    three = [dict(item(k, "dev"), exam="ooddev_v1") for k in ("c", "d", "k")]
    S.score = raising
    try:
        with as_root(root):
            got = None
            try:
                run(three, task="t_sys")
            except Z.ZooSystemic as e:
                got = e
    finally:
        S.score = orig
    check("the same exception on three models stops the task (ZooSystemic: exit 1)",
          got is not None and "planted failure" in str(got), got)

    print("the deadline")
    with T.env(INC_ZOO_TASK_DEADLINE_S="0"):
        with as_root(root):
            rd = run([dict(items[0], exam="imageweeds"), dict(items[2], exam="test")], task="t_dl")
    check("task_deadline_s 0: every item not_scored_time, the task ends normally", rd["counts"] ==
          {"not_scored_time": 2} and rd["status"] == "done", rd["counts"])

    print("the chain's GPU-hour cap and a task's allowance")
    ld = Z.zoo_dir(V) / "ledger"
    ld.mkdir(parents=True, exist_ok=True)
    (ld / "a__earlier_attempt_0.json").write_text(json.dumps({"format": Z.LEDGER_FORMAT, "kind": "a",
                                                             "job": "earlier", "task": "0", "started_s": 0.0,
                                                             "updated_s": 39.5 * 3600, "ended_s": 39.5 * 3600}))
    c, csha = T.conf()
    out = io.StringIO()
    with as_root(root), contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        rcap = Z.run_items(V, c, csha, [dict(items[0], exam="ooddev_v1", predicted_s=3600.0)], "t_cap", "a",
                           shard_predicted_s=3600.0, cap_h=39.0)
    check("a resubmission's spend counts with every earlier attempt's: past the cap, items are not_scored_budget, "
          "nothing scored", rcap["counts"] == {"not_scored_budget": 1} and "cap" in rcap.get("stopped", ""),
          (rcap["counts"], rcap.get("stopped")))
    (ld / "a__earlier_attempt_0.json").unlink()
    c2 = json.loads(json.dumps(c))
    c2["stages"]["task_allowance_min_s"] = 0
    with as_root(root), contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        ral = Z.run_items(V, c2, csha, [dict(items[0], exam="ooddev_v1")], "t_allow", "a", shard_predicted_s=0.001,
                          cap_h=39.0)
    check("a task past twice its shard's price stops (not_scored_budget: the task's allowance)",
          ral["counts"] == {"not_scored_budget": 1} and "allowance" in ral.get("stopped", ""), ral)

    print("the root checks")
    e_root = None
    try:
        Z.check_root(V)                       # INC_DIR is the INC tree, not a zoo root
    except Z.ZooSystemic as e:
        e_root = e
    lp = root / "splits" / "v1" / "LOCK.json"
    fake = TMP / "fakeroot"
    shutil.copytree(root, fake, symlinks=True)
    (fake / "splits" / "v1" / "LOCK.json").write_text(lp.read_text().replace("zoo-lock", "zoo-lockX"))
    e_lock = None
    with as_root(fake):
        try:
            Z.check_root(V)
        except Z.ZooSystemic as e:
            e_lock = e
    check("INC_DIR not a zoo root, or a LOCK other than the canonical one: systemic (exit 1)",
          e_root is not None and e_lock is not None, (e_root, e_lock))
    with as_root(root):
        rs = run([dict(items[4], exam="imageweeds", file_sha256="0" * 64)], task="t_sha")
    k_s, er_s = Z.record_of(V, items[4]["model_id"], "imageweeds")
    check("a file that does not hash as planned: an error record", rs["counts"] == {"error": 1} and
          "does not hash as planned" in (er_s or {}).get("error", ""), (rs["counts"], er_s))

    print("exam-root")
    local = TMP / "local"
    local.mkdir(exist_ok=True)
    sh = {"format": Z.SHARD_FORMAT, "stage": "a", "index": 7, "exams": ["dev"],
          "items": [dict(items[4], exam="imageweeds")], "predicted_s": 1.0}
    sh["exams"] = ["imageweeds"]
    Z._write_json(Z.shard_path(V, "a", 7), sh)
    with T.env(LOCAL=str(local), SLURM_ARRAY_JOB_ID="55", SLURM_ARRAY_TASK_ID="7", SLURM_JOB_ID="56"):
        lr, _o = T.quiet(Z.exam_root_cmd, V, "a", 7)
    lr = pathlib.Path(lr)
    imgs = list((lr / "exams" / "v1" / "imageweeds" / "images").iterdir())
    check("a node-local root: the shard's exam copied as real files beside the LOCK and the marker",
          str(lr).startswith(str(local)) and imgs and not any(p.is_symlink() for p in imgs)
          and (lr / Z.ROOT_MARKER).is_file(), (lr, len(imgs)))
    with as_root(lr):
        Z.check_root(V)
        rl = run(sh["items"], task="t_local")
    kl, recl = Z.record_of(V, items[4]["model_id"], "imageweeds")
    with as_root(root):
        dl = S.score(items[4]["file"], "imageweeds", TMP / "direct_iw.json", imgsz=64, batch=8, device="cpu")
    check("a record scored on the node-local root says exam_root local and equals the Lustre root's score",
          rl["counts"] == {"scored": 1} and recl["exam_root"] == "local"
          and abs(recl["result"]["agnostic_map50_95"] - dl["agnostic_map50_95"]) <= 1e-9,
          (rl["counts"], recl.get("exam_root")))
    import collections
    Usage = collections.namedtuple("Usage", "total used free")
    orig_du = shutil.disk_usage
    shutil.disk_usage = lambda p: Usage(10, 10, 0)
    try:
        with T.env(LOCAL=str(local)):
            lr2, _o = T.quiet(Z.exam_root_cmd, V, "a", 7)
    finally:
        shutil.disk_usage = orig_du
    check("too little room on $LOCAL: the Lustre root", lr2 == str(root), lr2)

    print("score --item")
    one = items[3]
    p1 = Z.score_path(V, one["model_id"], one["exam"])
    first = json.loads(p1.read_text())
    p1.unlink()
    with as_root(root):
        T.quiet(Z.score_cmd, V, None, None, False, "%s:%s" % (one["model_id"], one["exam"]))
    second = json.loads(p1.read_text())
    check("score --item ID:EXAM gives the shard's record", abs(first["result"]["agnostic_map50_95"] -
                                                               second["result"]["agnostic_map50_95"]) <= 1e-9
          and second["task"].startswith("item_"))
    print("an INC row is read on dev alone")
    e1 = item("e1", "dev")
    refusals = {}
    for what, call in (("score --item <e1>:test", lambda: Z.score_cmd(V, None, None, False, "%s:test" % e1["model_id"])),
                       ("score --item <e1>:test_v1", lambda: Z.score_cmd(V, None, None, False,
                                                                         "%s:test_v1" % e1["model_id"]))):
        with as_root(root):
            try:
                T.quiet(call)
                refusals[what] = None
            except Z.ZooRefused as e:
                refusals[what] = str(e)
    Z._write_json(Z.shard_path(V, "a", 8), {"format": Z.SHARD_FORMAT, "stage": "a", "index": 8,
                                            "exams": ["dev", "imageweeds"],
                                            "items": [dict(items[4], exam="dev"), dict(e1, exam="imageweeds")],
                                            "predicted_s": 2.0})
    with as_root(root):
        try:
            T.quiet(Z.score_cmd, V, "a", 8)
            refusals["a shard holding e1 on ImageWeeds"] = None
        except Z.ZooRefused as e:
            refusals["a shard holding e1 on ImageWeeds"] = str(e)
    Z.shard_path(V, "a", 8).unlink()
    tasks = sorted(p.name for p in (Z.zoo_dir(V) / "tasks").glob("*.json")
                   if p.name.startswith("item_%s" % e1["model_id"][:12]) or p.name == "a_008.json")
    check("an INC row (e1) on test, test v1, or in a shard beside another row's item: refused before anything is "
          "scored (no record, no task record)", all(v and "dev alone" in v for v in refusals.values())
          and all(Z.record_of(V, e1["model_id"], x)[0] is None for x in ("test", "test_v1", "imageweeds"))
          and Z.record_of(V, items[4]["model_id"], "dev")[0] == "scored" and not tasks, (refusals, tasks))
    with as_root(root):
        T.quiet(Z.score_cmd, V, None, None, False, "%s:dev" % e1["model_id"])
    check("  its dev item is scored (score --item <e1>:dev)", Z.record_of(V, e1["model_id"], "dev")[0] == "scored",
          Z.record_of(V, e1["model_id"], "dev")[0])
    check("nothing under the INC run directories changed; no score file appeared there",
          tree_hash(C.INC_DIR / "zt_inc") == inc_before
          and sorted(p.name for p in (C.INC_DIR / "zt_inc").glob("runs/*/scores/*.json")) == ["dev.json", "test.json"])
    print("\n%d failure(s)" % len(FAILURES))
    shutil.rmtree(TMP, ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
