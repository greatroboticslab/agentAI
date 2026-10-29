#!/usr/bin/env python3
"""DS1 (docs/CONTINUOUS_LOOP.md, Amendment 2026-10-29; docs/prereg/DS1_2026-10-29.md) on the stream ticker
(campaign.tick, the executor, the real stream_remote verbs on a fake cluster: tests/test_stream_ap_world.py).

- the levers: L30 (data steps), L31 (sheets, on the lab), L32 (1.2 M builds), L33 (scores) and L34 (verdicts) on
  MAINT, L32X (the long path) on TRAIN, every one record only; their prices and caps (DS.9.2), argv, the executor's
  reading of them and the cluster's grammar and job names; the evidence allow-list (DS1's records, never its
  external records, reports, sealed reads or INC_DIR/ds1);
- the gate (DS.9.3): no DS1 item while the deployed tree lacks the amendment, the cutter lacks DS1's never-train
  list, or the time-stamp record is missing; each a card; all three in place: the phase-1 screen first, from the
  real derived status of an empty DS1 tree;
- phase 1 end to end on a fake cluster that moves DS1's derived status as DS1's code would: the screen, the sheets
  rendered on the lab, MD-S0's draft, power and freeze, pools and draws after the sheets are decided, the
  manifests, the builds once RA is recorded (S1's builds once MT-W v2 is frozen), S0x's smoke and seeds on TRAIN,
  every score, then the phase verdict and its card naming the sealed reads; a card waiting for a person holds no
  lane (DS1's next item and the stream's own items go on);
- storage: a phase-1 item within phase 1's 60 GB keeps V3F's floor; past it lane DATA's floor too (the item waits,
  one card, the next item is proposed);
- the family cap (260 GPU-h) and the phase reserve's line (174.2): an item that would pass it is recorded 'not run
  (family cap)' with one card and never retried; S2 and S1c-R are cut first while the long pair's estimate is left;
- a failure is one card and one more proposal under a new id; after a second failure 'failed', no lane failure;
- an unknown outcome is followed by the job's name and DS1's record;
- the pure plan (ds1_plan): the date substitution (S1Ux in place of S2x, verdict v2 on its arms), an arm over the
  loader's line moved to the long path, KD/KT only once the SIU source is pooled, the verdict waiting for its arms.

Run:  python3 tests/test_stream_ap_ds1.py      (DS1_WORLD=static,gate,path,storage,cap,fail,lost,plan runs those)
"""
import copy
import json
import os
import pathlib
import shutil
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_world as W  # noqa: E402

from test_stream_ap_world import AP, DS, E, LS, NAME, S, SR, X, check  # noqa: E402

from weed_optimizer_framework.tools.inc2 import ds1 as D  # noqa: E402

FAILURES = W.FAILURES


def section(title):
    print("\n== %s" % title)


REPO_ROOT = W.PKG_ROOT.parent
DOM = LS.load_domain("weed")
CFG = DOM["ds1"]
MODT = "weed_optimizer_framework.tools."
K_SOURCE = CFG["k_source"]
IMAGES = {"S0": 39500, "KF": 3000, "KD": 3000, "KT": 3000, "S1CR": 52000, "S1C": 52000, "S1U": 60000, "S2": 75000}

# ---- DS1's derived status on the fake cluster: the real one (stream_remote.ds1_status) unless the world sets its own
REAL_STATUS = SR.ds1_status
STATUS = {}


def _status(inc):
    got = STATUS.get(str(inc))
    if got is None:
        return REAL_STATUS(inc)
    out = copy.deepcopy(got)
    out["timestamp"] = D.timestamp_ok(pathlib.Path(inc) / "ds1" / "v1" / "timestamp.json")
    return out


SR.ds1_status = _status


def ds1_world(tag, amendment=True, never_cut=True, timestamp=True, fake=True):
    """R0 complete (ready_r0; V3F's gate stays closed) and DS1's gate as asked: the amendment's file in the deployed
    tree (the lab's git root), the cutter's never-cut change (the branch's inc2/step1_stream.py, or a cutter without
    it), the time-stamp record naming the prereg's sha256 and two proofs. fake: the cluster's DS1 status is the
    world's dict (w.ds1, from the real status of the empty tree), moved by ds1_cluster as DS1's code would."""
    w = W.World(tag)
    (w.scripts / "run_inc2_long.sh").write_text("#!/bin/bash\n")
    w.ready_r0()
    if amendment:
        for f in [CFG["amendment_file"]] + [r["file"] for r in CFG.get("revisions") or []]:
            dst = w.dir / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(REPO_ROOT / f), str(dst))
    nc = w.lab / CFG["never_cut"]["module"]
    nc.parent.mkdir(parents=True, exist_ok=True)
    if never_cut:
        shutil.copyfile(str(W.PKG_ROOT / CFG["never_cut"]["module"]), str(nc))
    else:
        nc.write_text("# the cutter before DS1: it reads no DS1 list\n")
    if timestamp:
        w._w("ds1/v1/timestamp.json", {"format": D.FMT_TIMESTAMP, "sha256": D.PREREG_SHA256, "proofs": [
            {"file": "DS1_2026-10-29.md.ots", "sha256": "a" * 64},
            {"file": "DS1_2026-10-29.stamp.txt.ots", "sha256": "b" * 64}], "revisions": [
            {"file": r["file"], "sha256": r["sha256"], "proofs": [{"file": "c", "sha256": "c" * 64},
                                                                 {"file": "d", "sha256": "d" * 64}]}
            for r in D.REVISIONS]})
    else:
        (w.inc / "ds1" / "v1").mkdir(parents=True, exist_ok=True)
    w.ds1 = REAL_STATUS(w.inc) if fake else None
    STATUS[str(w.inc)] = w.ds1
    w.ds1_enlarge = False
    w.ds1_lab_done = set()
    return w


def source(**kw):
    out = {"decided": False, "decision": None, "audit_required": False, "audited": False, "ready": False,
           "why": "no decision", "pooled": False, "md_n": False, "mtw_draft": False, "mtw": False, "checked": False,
           "quarantined_whole": False, "role_c": False, "settled_out": False, "exam_draw": True}
    out.update(kw)
    return out


def summary(w, phase, step):
    w._w("splits/ds1/summary.json", {"format": "inc2-ds1-data/1", "status": "index", "arms": {}, "exams": {},
                                     "last_step": {"phase": int(phase), "step": step, "utc": W.utc(w.t[0])}})


def data_step(w, phase, step):
    """What inc2.ds1_data build --phase P --step S changes in the derived status (and the summary's last step)."""
    st = w.ds1
    ph = st["phases"][str(phase)]
    sh = st["sheets"]
    md = st["md_s0"]
    if step == "screen":
        ph["revisions"] += 1
        ph["new_batches"] = 0
        names = ("src_alpha", "src_beta", K_SOURCE) if phase == 1 else ("src_gamma",)
        for s_ in names:
            # the K source (SIU) is never drawn into an exam (ds1_data.plan_state: exam_draw False)
            ph["sources"].setdefault(s_, source(exam_draw=s_ != K_SOURCE))
        sh["source_batches"] += 1
        if phase == 1 and ph["revisions"] == 1:
            sh["ln_batches"], sh["mtw_batches"] = 1, sh["mtw_batches"] + 1
    elif step == "md_s0":
        if not md["drafts"] or (md["power"] and md["enlarge"]):
            md["drafts"] += 1
            md.update(draft="draft_r%03d" % md["drafts"], power=False, enlarge=None)
        elif md["power"] and not md["enlarge"]:
            md["frozen"] = True
    elif step == "pools":
        for x in ph["sources"].values():
            if x["ready"]:
                x["pooled"] = True
    elif step == "draw":
        for x in ph["sources"].values():
            if x["pooled"] and x["decision"] == "accept" and x.get("exam_draw", True) and not x["md_n"]:
                x.update(md_n=True, mtw_draft=True)
                sh["mtw_batches"] += 1
            elif x["checked"] and x["mtw_draft"] and not x["mtw"]:
                x["mtw"] = True
    elif step == "manifests":
        arms = st["arms"]
        if phase == 1:
            for a in ("S0", "KF"):
                arms.setdefault(a, {"images": IMAGES[a], "sha256": a.lower() * 4, "over_line": False})
            if (ph["sources"].get(K_SOURCE) or {}).get("pooled"):
                for a in ("KD", "KT"):
                    arms.setdefault(a, {"images": IMAGES[a], "sha256": a.lower() * 4, "over_line": False})
            if all(x["decided"] and (x["decision"] == "reject" or (x["pooled"] and (
                    x["decision"] != "accept" or x["md_n"] or not x.get("exam_draw", True))))
                   for x in ph["sources"].values()):
                for a in ("S1CR", "S1C", "S1U"):
                    arms.setdefault(a, {"images": IMAGES[a], "sha256": a.lower() * 4, "over_line": False})
        else:
            arms.setdefault("S2", {"images": IMAGES["S2"], "sha256": "s2" * 4, "over_line": False})
    summary(w, phase, step)


def verdict_doc(phase):
    return {"format": "inc2-ds1-verdict/1", "status": "decided", "phase": phase, "decided_on": "dev+md",
            "primaries": {"P1": {"status": "decided", "x": "S1C", "y": "S0", "diff": 0.012, "z": 2.3,
                                 "threshold": 2.128, "interval_90": [0.004, 0.02], "reading": "confirmed"}},
            "reads": {"order": [{"arm": "S0X", "exam": "mtw_v2", "kind": "S0x on MT-W"}]}}


def ds1_cluster(w, skip=()):
    """The cluster's side of DS1's jobs: each queued DS1 job ends now and moves the status as its code would."""
    exps = {v: k for k, v in CFG["build_exps"].items()}
    for j in list(w.squeue):
        n = j["name"]
        if n in skip or not any(j is x for x in w.squeue):
            continue
        if n.startswith("inc_build_ds1_data_p"):
            phase, step = int(n[len("inc_build_ds1_data_p")]), n[len("inc_build_ds1_data_p1_"):]
            w.job_done(n)
            data_step(w, phase, step)
        elif n.startswith("inc_build_ds1_score_"):
            arm = n[len("inc_build_ds1_score_"):].upper()
            w.job_done(n)
            w.ds1["scores"][arm] = "complete"
            if arm == "POWER":
                w.ds1["md_s0"].update(power=True, enlarge=bool(w.ds1_enlarge))
                if not w.ds1_enlarge:
                    w.ds1["power_final"] = True
                w.ds1_enlarge = False
            w._w("capacity/ds1_score_%s.json" % arm.lower(), {"format": "inc2-ds1-score-record/1", "arm": arm,
                                                             "status": "complete", "exam": "dev"})
        elif n.startswith("inc_build_ds1_v"):
            phase = int(n[len("inc_build_ds1_v"):])
            w.job_done(n)
            w.ds1["verdicts"][str(phase)] = {"status": "decided", "reads": [{"arm": "S0X", "exam": "mtw_v2"}],
                                             "substitution": None}
            w._w("capacity/ds1_v%d.json" % phase, verdict_doc(phase))
        elif n.startswith("inc_long_ds1_"):
            arm, what = n[len("inc_long_ds1_"):].rsplit("_", 1)
            arm = arm.upper()
            w.job_done(n)
            la = w.ds1["long"]["arms"][arm]
            if what == "smoke":
                la.update(smoke="complete", projection_h=17.2, walltime_h=24.0)
                if arm == "S0X" and not w.ds1["long"]["pair"]:
                    w.ds1["long"]["pair"] = {"image_epochs": D.PAIR_IMAGE_EPOCHS, "decided_by": "S0X"}
            else:
                la["trained"] = sorted(set(la["trained"]) | {int(what[1:])})
        elif n.startswith("inc_build_") and n[len("inc_build_"):] in exps:
            exp = n[len("inc_build_"):]
            w.experiment(exp, final=[w.final_row("base", 0.85, 0.002, 3)])
            w.job_done(n)


def ds1_lab(w):
    """L31's lab jobs end (each pushes its render record to the cluster: the derived status lists it rendered)."""
    for l in w.runner.launched:
        if not l["job"].startswith("ds1sheets_") or l["job"] in w.ds1_lab_done:
            continue
        a = l["argv"]
        kind, batch = a[a.index("--kind") + 1], int(a[a.index("--batch") + 1])
        w.runner.results[l["job"]] = {"ok": True, "rc": 0, "seconds": 3.0}
        w.ds1_lab_done.add(l["job"])
        if w.ds1 is not None:
            w.ds1["sheets"]["rendered"] = sorted(set(w.ds1["sheets"]["rendered"]) | {"%s_%03d" % (kind, batch)})


def run(w, ticks=40, until=None, skip=()):
    for _ in range(ticks):
        w.tick(1)
        ds1_lab(w)
        ds1_cluster(w, skip=skip)
        if until and until():
            return True
    return False


def pro(w, lever=None):
    return [e for e in w.events("proposed") if e.get("lever") in ((lever,) if lever else S.DS1_LEVERS)]


def item_of(e):
    """What a DS1 proposal names: L30 phase:step, L31 kind:batch, L32 the arm, L32X arm:verb, L33 the arm, L34 the
    phase."""
    a = e.get("argv") or []
    lv = e.get("lever")

    def f(flag):
        return a[a.index(flag) + 1] if flag in a else None
    if lv == "L30":
        return "%s:%s" % (f("--phase"), f("--step"))
    if lv == "L31":
        return "%s:%s" % (f("--kind"), f("--batch"))
    if lv == "L32":
        return f("--ds1")
    if lv == "L32X":
        return "%s:%s" % (f("--arm"), "smoke" if "smoke" in a else "train")
    if lv == "L33":
        return f("--arm")
    return "v%s" % f("--phase")


def seq(w):
    return [(e.get("lever"), item_of(e)) for e in pro(w)]


def items(w):
    return ((w.state() or {}).get("ds1") or {}).get("items") or {}


def cards(w, needle=None):
    return [c for c in (w.state() or {}).get("cards") or [] if needle is None or needle in c["title"]]


def decide(w, phase, decisions, audit=()):
    """The delegate's sheets recorded (inc2.ds1_sheets record): each source decided; audit: the owner's audit
    applies and is not yet done."""
    for s_, d in decisions.items():
        x = w.ds1["phases"][str(phase)]["sources"][s_]
        x.update(decided=True, decision=d, ready=d != "reject" and s_ not in audit, why="" if d != "reject" else
                 "rejected", audit_required=s_ in audit, audited=False, settled_out=d == "reject")
    w.ds1["sheets"]["audit_pending"] = sorted(audit)


def state_write(w, st):
    S.StreamPaths(str(w.lab), "weed").state(NAME).write_text(json.dumps(st))


# ------------------------------------------------------------------ the levers, prices, argv and grammar
def t_static():
    section("DS1's levers: L30, L31, L32, L33, L34 on MAINT and L32X on TRAIN, record only; prices and caps; argv; "
              "the cluster's grammar; the evidence")
    check("the lanes: L32X on TRAIN, the five others on MAINT; L31 runs on the lab; L30, L31, L32X, L33 and L34 "
          "record only (an unknown outcome followed by its job name and record), L32 followed by its experiment as "
          "V3F's builds; L30, L32, L32X, L33 and L34 envelope levers, their actions envelope actions",
          S.LANE_OF["L32X"] == "TRAIN" == LS.row("L32X")["lane"]
          and all(S.LANE_OF[l] == "MAINT" == LS.row(l)["lane"] for l in ("L30", "L31", "L32", "L33", "L34"))
          and all(l in S.RECORD_ONLY_LEVERS for l in S.DS1_LEVERS if l != "L32") and "L32" not in S.RECORD_ONLY_LEVERS
          and X.LAB_ACTIONS.count(LS.row("L31")["policy_action"]) == 1
          and all(l in LS.envelope_levers() for l in ("L30", "L32", "L32X", "L33", "L34"))
          and "L31" not in LS.envelope_levers()
          and all(X.STREAM_ENVELOPE_LEVERS.get(l) == (LS.row(l)["policy_action"],) for l in
                  ("L30", "L32", "L32X", "L33", "L34"))
          and all(LS.row(l)["policy_action"] in AP.ENVELOPE_ACTIONS for l in ("L30", "L32", "L32X", "L33", "L34")))
    check("the domain's ds1 block: the amendment's commit and sha256 (inc2.ds1's), the cap 260 GPU-h and the phase "
          "reserve 85.8, phase 1's 60 GB, two failures, the score arms inc2.ds1's",
          CFG["amendment_sha256"] == D.PREREG_SHA256 and CFG["amendment_commit"] == D.PREREG_COMMIT
          and CFG["cap_gpu_hours"] == 260.0 and CFG["phase_reserve_gpu_hours"] == 85.8
          and CFG["phase1_bytes_gb"] == 60.0 and CFG["max_failures"] == 2
          and sorted(CFG["score_arms"]) == sorted(D.SCORE_ARMS) == sorted(SR.DS1_SCORE_ARMS)
          and sorted(CFG["build_exps"]) == sorted(SR.DS1_BUILD_ARMS), CFG.get("score_arms"))
    prices = {}
    for lid, prm, info in (("L30", {"pkg": "inc2", "phase": 1, "step": "screen"}, {}),
                           ("L30", {"pkg": "inc2", "phase": 2, "step": "pools"}, {}),
                           ("L30", {"pkg": "inc2", "phase": 1, "step": "manifests"}, {}),
                           ("L32", {"pkg": "inc2", "exp": "ds1_s0_m640", "arm": "S0", "seeds": "0,1,2",
                                    "role": "baseline"}, {"images": 39500}),
                           ("L32", {"pkg": "inc2", "exp": "ds1_kd_m640", "arm": "KD", "seeds": "0,1,2",
                                    "role": "baseline"}, {"images": 3000}),
                           ("L32X", {"pkg": "inc2", "verb": "smoke", "arm": "S0X"}, {}),
                           ("L32X", {"pkg": "inc2", "verb": "train", "arm": "S0X", "seeds": "0,1,2"}, {}),
                           ("L32X", {"pkg": "inc2", "verb": "train", "arm": "S0X", "seeds": "1,2"},
                            {"projection_h": 17.2}),
                           ("L32X", {"pkg": "inc2", "verb": "train", "arm": "S2X", "seeds": "0", "walltime_h": 36},
                            {"projection_h": 25.0}),
                           ("L33", {"pkg": "inc2", "arm": "POWER"}, {}), ("L33", {"pkg": "inc2", "arm": "KD"}, {}),
                           ("L33", {"pkg": "inc2", "arm": "S1CR"}, {}), ("L34", {"pkg": "inc2", "phase": 2}, {})):
        est, det = LS.price(lid, LS.policy_params(lid, prm), DOM, info)
        ok, bad = LS.check_params(lid, dict(LS.policy_params(lid, prm), est_gpu_hours=est))
        prices["%s %s" % (lid, prm.get("step") or (prm.get("arm") if lid != "L32X" else "%s %s %s" % (
            prm["arm"], prm["verb"], prm.get("seeds") or "")) or prm.get("phase"))] = (est, ok)
    def build(n):
        return round(3 * n * int(1.2e6 / n + 0.5) * 13.2 / 3.6e6 + 0.3, 3)
    want = {"L30 screen": 1.0, "L30 pools": 0.5, "L30 manifests": 0.75, "L32 S0": build(39500), "L32 KD": build(3000),
            "L32X S0X smoke ": 0.5, "L32X S0X train 0,1,2": 42.6, "L32X S0X train 1,2": 34.4,
            "L32X S2X train 0": 25.0, "L33 POWER": 1.5, "L33 KD": 0.8, "L33 S1CR": 0.6, "L34 2": 0.5}
    check("prices (DS.9.2): L30 per step (screen 1.0, pools 0.5, manifests 0.75), L32 3 seeds x N x round(1.2 M / N) "
          "epochs x 13.2 ms (DS.11: 4.4 GPU-h a 1.2 M run) + finals, the build job settled, not estimated (S0 %.3f, KD "
          "%.3f), L32X's smoke 0.5 and 14.2 per seed before a "
          "smoke, the smoke's projection after it (17.2 x 2), S2x's 36 h re-submission at its 25.0, L33 per arm, L34 "
          "0.5; each inside its policy row" % (build(39500), build(3000)),
          all(k in prices and abs(prices[k][0] - v) < 1e-6 and prices[k][1] for k, v in want.items()), prices)
    caps = {}
    for tag, lid, prm, info in (
            ("a seed over 19.2 h at 24 h", "L32X", {"pkg": "inc2", "verb": "train", "arm": "S0X", "seeds": "0"},
             {"projection_h": 19.5}),
            ("S2x's seed over 28.8 h at 36 h", "L32X", {"pkg": "inc2", "verb": "train", "arm": "S2X", "seeds": "0",
                                                        "walltime_h": 36}, {"projection_h": 29.0}),
            ("a data step above its cap", "L30", {"pkg": "inc2", "phase": 1, "step": "screen"}, {})):
        dom = DOM if tag != "a data step above its cap" else dict(DOM, ds1=dict(CFG, costs=dict(
            CFG["costs"], data_cap=dict(CFG["costs"]["data_cap"], screen=0.5))))
        try:
            LS.price(lid, LS.policy_params(lid, prm), dom, info)
            caps[tag] = False
        except LS.LeverError:
            caps[tag] = True
    check("  an estimate above its cap is refused: %s" % "; ".join(caps), all(caps.values()), caps)
    bad = [LS.check_params("L33", LS.policy_params("L33", {"pkg": "inc2", "arm": "S9", "est_gpu_hours": 1.0}))[0],
           LS.check_params("L30", LS.policy_params("L30", {"pkg": "inc2", "phase": 3, "step": "screen",
                                                           "est_gpu_hours": 1.0}))[0],
           LS.check_params("L30", LS.policy_params("L30", {"pkg": "inc2", "phase": 1, "step": "train",
                                                           "est_gpu_hours": 1.0}))[0]]
    check("  an arm, phase or step outside DS1's lists is outside the policy rows", not any(bad), bad)
    for lid, prm, tail, kind in (
            ("L30", {"pkg": "inc2", "phase": 1, "step": "screen"},
             [MODT + "inc2.ds1_data", "build", "--phase", "1", "--step", "screen"], "build"),
            ("L32", {"pkg": "inc2", "exp": "ds1_s1c_m640", "arm": "S1C", "seeds": "0,1,2", "role": "baseline"},
             [MODT + "inc2.baseline", "build", "--exp", "ds1_s1c_m640", "--ds1", "S1C", "--seeds", "0,1,2", "--role",
              "baseline"], "build"),
            ("L32X", {"pkg": "inc2", "verb": "smoke", "arm": "S0X"}, [MODT + "inc2.ds1_long", "smoke", "--arm", "S0X"],
             "long"),
            ("L32X", {"pkg": "inc2", "verb": "train", "arm": "S2X", "seeds": "1,2", "walltime_h": 36},
             [MODT + "inc2.ds1_long", "train", "--arm", "S2X", "--seeds", "1,2", "--walltime-h", "36"], "long"),
            ("L33", {"pkg": "inc2", "arm": "K0"}, [MODT + "inc2.ds1", "score", "--arm", "K0"], "build"),
            ("L34", {"pkg": "inc2", "phase": 1, "cap_refused": "build:S1CR", "ticker_failed": "score:KD"},
             [MODT + "inc2.ds1", "verdict", "--phase", "1", "--cap-refused", "build:S1CR", "--ticker-failed",
              "score:KD"], "build")):
        pp = LS.policy_params(lid, prm)
        argv = LS.render(lid, pp)
        act = LS.row(lid)["policy_action"]
        back = X.params_from_argv(act, argv)
        rend = X.render(act, pp, {"child_exp": prm.get("exp")} if lid == "L32" else {})
        check("%s renders ... %s; the executor reads it back and sends stream-submit %s" % (
            lid, " ".join(tail), kind), argv[-len(tail):] == tail and back == pp
            and rend["remote"][:2] == ["stream-submit", kind] and X.STREAM_REMOTE[act] == kind,
            (argv, back, rend["remote"]))
    pp = LS.policy_params("L31", {"kind": "source", "batch": 2, "lab_inc": "/lab/r/results/framework/inc"})
    check("L31 renders python -m ...inc2.ds1_sheets render --kind source --batch 2 --lab-inc <the lab's INC_DIR> "
          "(a lab action: the lab's process)", LS.render("L31", pp)[1:] == [
              "-m", MODT + "inc2.ds1_sheets", "render", "--kind", "source", "--batch", "2", "--lab-inc",
              "/lab/r/results/framework/inc"], LS.render("L31", pp))
    for kind, args, want_ok, names in (
            ("build", ["inc2.ds1_data", "build", "--phase", "1", "--step", "screen"], True,
             "inc_build_ds1_data_p1_screen"),
            ("build", ["inc2.ds1_data", "build", "--phase", "2", "--step", "manifests"], True,
             "inc_build_ds1_data_p2_manifests"),
            ("build", ["inc2.ds1_data", "build", "--phase", "3", "--step", "screen"], False, None),
            ("build", ["inc2.ds1_data", "build", "--phase", "1", "--step", "freeze"], False, None),
            ("build", ["inc2.ds1_data", "rehearse"], False, None),
            ("build", ["inc2.ds1", "score", "--arm", "POWER"], True, "inc_build_ds1_score_power"),
            ("build", ["inc2.ds1", "score", "--arm", "S1UX"], True, "inc_build_ds1_score_s1ux"),
            ("build", ["inc2.ds1", "score", "--arm", "S3"], False, None),
            ("build", ["inc2.ds1", "verdict", "--phase", "2"], True, "inc_build_ds1_v2"),
            ("build", ["inc2.ds1", "verdict", "--phase", "1", "--cap-refused", "build:S1CR,long:S2X:train"], True,
             "inc_build_ds1_v1"),
            ("build", ["inc2.ds1", "verdict", "--phase", "1", "--cap-refused", "verdict:1"], False, None),
            ("build", ["inc2.ds1", "score-test", "--arm", "S0", "--exam", "test_v1"], False, None),
            ("build", ["inc2.ds1", "test-read", "--arm", "S0", "--exam", "test_v1"], False, None),
            ("build", ["inc2.ds1", "test-report", "--arm", "S0", "--exam", "test_v1"], False, None),
            ("build", ["inc2.ds1", "ra"], False, None),
            ("build", ["inc2.ds1", "timestamp"], False, None),
            ("build", ["inc2.baseline", "build", "--exp", "ds1_s0_m640", "--ds1", "S0", "--seeds", "0,1,2", "--role",
                       "baseline"], True, "inc_build_ds1_s0_m640"),
            ("build", ["inc2.baseline", "build", "--exp", "ds1_s0_m640", "--ds1", "S0", "--seeds", "0", "--role",
                       "baseline"], False, None),
            ("build", ["inc2.baseline", "build", "--exp", "ds1_s0_m640", "--ds1", "S0X", "--seeds", "0,1,2", "--role",
                       "baseline"], False, None),
            ("build", ["inc2.baseline", "build", "--exp", "ds1_s0_m640", "--ds1", "S0", "--seeds", "0,1,2", "--role",
                       "baseline", "--manifest", "/x/m.jsonl"], False, None),
            ("long", ["inc2.ds1_long", "smoke", "--arm", "S0X"], True, "inc_long_ds1_s0x_smoke"),
            ("long", ["inc2.ds1_long", "train", "--arm", "S0X", "--seeds", "0,1,2"], True, "inc_long_ds1_s0x_s0"),
            ("long", ["inc2.ds1_long", "train", "--arm", "S2X", "--seeds", "2", "--walltime-h", "36"], True,
             "inc_long_ds1_s2x_s2"),
            ("long", ["inc2.ds1_long", "train", "--arm", "S0X", "--seeds", "2,1"], False, None),
            ("long", ["inc2.ds1_long", "train", "--arm", "S0X", "--seeds", "3"], False, None),
            ("long", ["inc2.ds1_long", "train", "--arm", "S0X", "--seeds", "0", "--walltime-h", "48"], False, None),
            ("long", ["inc2.ds1_long", "status"], False, None),
            ("long", ["inc2.v3fam_detr", "smoke"], False, None)):
        try:
            req = SR.parse_submit(kind, args)
            got, jn = True, SR.job_name(req, {})
        except W.R.Refused:
            got, jn = False, None
        check("the cluster's %s grammar %s %s%s" % (kind, "admits" if want_ok else "refuses", " ".join(args),
                                                    " (job %s)" % names if names else ""),
              got == want_ok and (names is None or jn == names), (got, jn))
    check("the evidence allow-lists DS1's verdicts, score records, power pre-flight, RA, data index and derived status; "
          "never its external records, reports, sealed reads or INC_DIR/ds1's files",
          all(E.allowed(x) for x in ["capacity/ds1_v1.json", "capacity/ds1_v2.json", "capacity/ds1_power.json",
                                     "capacity/ds1_ra_gate.json", "splits/ds1/summary.json", "ds1/v1/status.json"]
              + ["capacity/ds1_score_%s.json" % a.lower() for a in D.SCORE_ARMS])
          and not any(E.allowed(x) for x in ("capacity/ds1_v1_external.json", "capacity/ds1_v2_external.json",
                                             "capacity/ds1_test_s0_test_v1.json", "capacity/ds1_report.json",
                                             "capacity/ds1_score_power_external.json", "ds1/v1/timestamp.json",
                                             "ds1/v1/reads/S0X/mtw_v2/read.json", "ds1/v1/power/draft_r001.json",
                                             "splits/ds1/never_train.jsonl", "splits/ds1/md_v1/md_s0/frozen.json"))
          and "ds1" in E.RESERVED_DIRS)
    # the domain-free rule (S13): no stream module names DS1's source or a stream id
    base = W.PKG_ROOT / "weed_optimizer_framework" / "tools" / "inc_autopilot"
    hits = [(n, i) for n in ("stream.py", "stream_remote.py", "diagnose_stream.py", "levers_stream.py", "executor.py",
                             "evidence.py")
            for i, line in enumerate((base / n).read_text().splitlines(), 1) if "ds1" in line.lower()
            and any(t in line for t in [K_SOURCE, "SIU", "siu"] + list(CFG["reaudit_segments"]))]
    check("no DS1 line of the stream modules names DS1's K source, SIU or the segments to re-audit (the domain's ds1 "
          "block names them; S13's scan covers every other domain term)", not hits, hits)


# ------------------------------------------------------------------ the gate
def t_gate():
    section("DS1's gate (DS.9.3): the amendment, the cutter's never-cut change, the time stamp")
    res = {}
    for tag, kw in (("no amendment", {"amendment": False}), ("a cutter without DS1's list", {"never_cut": False}),
                    ("no time stamp", {"timestamp": False})):
        w = ds1_world("ds1_gate_%s" % tag.split()[1], **kw)
        w.tick(3)
        res[tag] = ([e.get("lever") for e in pro(w)], [c["title"] for c in cards(w, "DS1")])
    check("no DS1 lever while %s" % "; ".join(res), all(not v[0] for v in res.values()), res)
    check("  a card names the cutter's change and the time-stamp record (once each); none without the amendment",
          res["a cutter without DS1's list"][1] == ["DS1 waits for the cutter's never-cut change"]
          and "DS1 waits for its time-stamp record" in res["no time stamp"][1]
          and not res["no amendment"][1], res)
    wrong = ds1_world("ds1_gate_sha")
    (wrong.dir / CFG["amendment_file"]).write_text("an edited amendment\n")
    wrong.tick(3)
    check("  a deployed amendment whose sha256 differs: no DS1 lever", not pro(wrong), seq(wrong))
    w = ds1_world("ds1_gate_open", fake=False)
    w.tick(2)
    first = pro(w)
    cites = sorted(x.get("pointer") for x in (first[0] if first else {}).get("cites") or [])
    ex = [e for e in w.events("executed") if e.get("lever") in S.DS1_LEVERS]
    check("all three in place, the cluster's real status of an empty DS1 tree: L30 phase 1 screen first, one job "
          "inc_build_ds1_data_p1_screen, executed within the envelope, citing the lock and the gate",
          seq(w)[:1] == [("L30", "1:screen")]
          and [x["name"] for x in w.submits if "ds1_data" in " ".join(x["argv"])] == ["inc_build_ds1_data_p1_screen"]
          and ex and ex[0].get("basis") == "envelope"
          and {"/stage/lock", "/stage/ds1/amendment", "/stage/ds1/never_cut", "/stage/ds1/timestamp"} <= set(cites),
          (seq(w), cites, [e.get("basis") for e in ex]))
    ws = ds1_world("ds1_gate_screened", fake=False)
    from weed_optimizer_framework.tools.inc2 import ds1_data as DDm
    ws._w("splits/ds1/screen/p1_r001.json", {"format": DDm.FMT_SCREEN, "phase": 1, "revision": 1, "candidates": 0,
                                             "live": 0, "sources": {}, "inputs": {"step1_batches": []}})
    ws.tick(2)
    check("  the real status reads the snapshot's INC_DIR: with phase 1 screened there (splits/ds1/screen/p1_r001.json), "
          "MD-S0's draft comes first, no second screen", seq(ws)[:1] == [("L30", "1:md_s0")], seq(ws))
    snap = json.loads(S.StreamPaths(str(w.lab), "weed").latest(NAME).read_text())
    arts = ((snap.get("stream") or {}).get("decision") or {}).get("artifacts") or snap.get("artifacts") or {}
    rec = items(w).get("data:1:screen") or {}
    check("  the snapshot ships the derived status (ds1/v1/status.json, stream_remote.ds1_status); DS1's record holds "
          "the screen running at its 1.0 GPU-h estimate, phase 1", (arts.get(SR.DS1_STATUS) or {}).get("format") ==
          "inc_autopilot.ds1_status/1" and rec.get("state") == "running" and rec.get("estimate") == 1.0
          and rec.get("phase") == 1, (sorted(k for k in arts if "ds1" in k), rec))


# ------------------------------------------------------------------ phase 1, end to end
def t_path():
    section("phase 1 end to end: screen, sheets, MD-S0, pools, draws, manifests, builds, the long path, scores, "
              "the verdict; a card waiting for a person holds no lane")
    w = ds1_world("ds1_path")
    run(w, ticks=8, until=lambda: w.ds1["phases"]["1"]["revisions"] == 1 and len(w.ds1["sheets"]["rendered"]) >= 3)
    run(w, ticks=6, until=lambda: len(w.ds1["sheets"]["rendered"]) >= 3)
    s1 = seq(w)
    lab = [l["argv"] for l in w.runner.launched if l["job"].startswith("ds1sheets_")]
    check("the screen, then the source, LN and MT-W v1 sheets rendered on the lab (L31, one batch each, the lab's "
          "INC_DIR)", s1[:1] == [("L30", "1:screen")] and sorted(x for l, x in s1 if l == "L31") == [
              "ln:1", "mtw:1", "source:1"] and len(lab) == 3
          and all(a[a.index("--lab-inc") + 1] == str(S.StreamPaths(str(w.lab), "weed").lab_inc) for a in lab), (s1, lab))
    run(w, ticks=3)
    wc = cards(w, "DS1 waits for a person")
    check("a card names what waits for a person (one card per content: RA alone before the screen, then RA, the "
          "sheets of phase 1 naming each source, the MT-W check, the LN sheet)",
          len(wc) == 2 and all(x in wc[-1]["detail"] for x in ("RA (DS.8.1", "sheets waiting (phase 1",
                                                                 "src_alpha", "MT-W check waiting", "the LN sheet",
                                                                 "a card holds no lane"))
          and "sheets waiting" not in wc[0]["detail"], [c["detail"][:400] for c in wc])
    lanes = (w.state() or {}).get("lanes") or {}
    check("  the card holds no lane: with it up, DS1's next item whose conditions hold (MD-S0's draft) is proposed "
          "and no lane has a hold", ("L30", "1:md_s0") in seq(w)
          and not any((x or {}).get("hold") for x in lanes.values()),
          (seq(w), {k: (x or {}).get("hold") for k, x in lanes.items()}))
    # MD-S0 goes on while the sheets wait: its draft, the power pre-flight (POWER), its freeze
    run(w, ticks=12, until=lambda: w.ds1["md_s0"]["frozen"])
    s2 = seq(w)
    check("while the person's steps wait, MD-S0 goes on: L30 md_s0 (its draft), L33 POWER, L30 md_s0 (its freeze); no "
          "pool, draw, manifest or build yet", w.ds1["md_s0"]["frozen"]
          and [x for x in s2 if x[0] in ("L30", "L33")][1:] == [("L30", "1:md_s0"), ("L33", "POWER"),
                                                                 ("L30", "1:md_s0")], s2)
    rq = cards(w, "re-audit")
    check("the re-audit card names the domain's two segments (DS.10 item 2)", len(rq) == 1 and all(
        s_ in rq[0]["title"] for s_ in CFG["reaudit_segments"]), [c["title"] for c in rq])
    # the delegate decides the sheets: alpha and the K source accepted, beta rejected
    decide(w, 1, {"src_alpha": "accept", K_SOURCE: "accept", "src_beta": "reject"})
    run(w, ticks=12, until=lambda: "S0" in w.ds1["arms"])
    s3 = seq(w)[len(s2):]
    check("the sheets decided: pools, draws, then the manifests (S0 and KF, KD and KT once the K source is pooled; S1's "
          "once every accepted source's MD-N is drawn)", [x for x in s3 if x[0] == "L30"][:3] == [
              ("L30", "1:pools"), ("L30", "1:draw"), ("L30", "1:manifests")]
          and sorted(w.ds1["arms"]) == sorted(["S0", "KF", "KD", "KT", "S1CR", "S1C", "S1U"]), (s3, sorted(w.ds1["arms"])))
    run(w, ticks=4)
    tw = [e for e in pro(w) if e.get("lever") in ("L32", "L32X")]
    dr = ((w.state().get("diagnosis") or {}).get("DR0") or {}).get("detail") or {}
    check("no build and no long-path job before RA is recorded (DS.9.3)", not tw, [item_of(e) for e in tw])
    w.ds1["ra"] = True
    run(w, ticks=30, until=lambda: items(w).get("build:KT", {}).get("state") in ("done", "running") and
        items(w).get("build:KT", {}).get("state") == "done")
    s4 = [x for x in seq(w) if x[0] == "L32"]
    check("RA recorded: the 1.2 M builds of S0, KF, KD, KT in the domain's order; S1c-R, S1c and S1u wait for MT-W v2's "
          "freeze (the MT-W check)", s4[:4] == [("L32", "S0"), ("L32", "KF"), ("L32", "KD"), ("L32", "KT")]
          and not [x for x in s4 if x[1] in ("S1C", "S1U", "S1CR")], s4)
    ln = [x for x in seq(w) if x[0] == "L32X"]
    check("  on TRAIN meanwhile: S0x's smoke, then its three seeds in one item (jobs inc_long_ds1_s0x_s0..s2)",
          ln[:2] == [("L32X", "S0X:smoke"), ("L32X", "S0X:train")]
          and sorted(x["name"] for x in w.submits if x["name"].startswith("inc_long_ds1_s0x")) == [
              "inc_long_ds1_s0x_s0", "inc_long_ds1_s0x_s1", "inc_long_ds1_s0x_s2", "inc_long_ds1_s0x_smoke"], ln)
    # the MT-W check: the person's check recorded, then the draw freezes MT-W v2 and S1's builds follow
    for s_ in ("src_alpha", K_SOURCE):
        w.ds1["phases"]["1"]["sources"][s_]["checked"] = True
    w.ds1["mtw_v1_1"] = True
    w.ds1["sheets"]["ln_recorded"] = True
    run(w, ticks=60, until=lambda: (w.ds1["verdicts"]["1"] or {}).get("status") == "decided")
    s5 = seq(w)
    builds = [x[1] for x in s5 if x[0] == "L32"]
    scores = [x[1] for x in s5 if x[0] == "L33"]
    check("the MT-W check recorded: the draw freezes MT-W v2, then S1c-R, S1c and S1u build",
          builds == ["S0", "KF", "KD", "KT", "S1CR", "S1C", "S1U"]
          and s5.index(("L30", "1:draw"), s5.index(("L30", "1:manifests"))) < s5.index(("L32", "S1CR")), builds)
    check("every phase-1 arm scored once (K0 once the power pre-flight is final), S0x once its seeds are trained; "
          "then L34 phase 1 once, with no --cap-refused or --ticker-failed",
          sorted(scores) == sorted(["POWER", "K0", "KD", "KT", "KF", "S0", "S1CR", "S1C", "S1U", "S0X"])
          and len(scores) == len(set(scores)) and [x for x in s5 if x[0] == "L34"] == [("L34", "v1")]
          and not any("--cap-refused" in (e.get("argv") or []) or "--ticker-failed" in (e.get("argv") or [])
                      for e in pro(w, "L34")), (scores, [x for x in s5 if x[0] == "L34"]))
    ex = [e for e in w.events("executed") if e.get("lever") in S.DS1_LEVERS]
    check("  every DS1 proposal executed once, the envelope levers within the envelope, L31 by its R0 row",
          len(ex) == len(pro(w)) and all(e.get("basis") == "envelope" for e in ex if e.get("lever") != "L31"),
          sorted({(e.get("lever"), e.get("basis")) for e in ex}))
    w.tick(2)
    vc = cards(w, "DS1's verdict v1 is decided")
    check("one verdict card: each primary's numbers and reading, each sealed read's exact commands in the verdict's "
          "order (test-read --read-by owner|delegate, INCAP_DECIDED_BY, test-report); nothing switches",
          len(vc) == 1 and "test-read --arm S0X --exam mtw_v2 --read-by owner|delegate" in vc[0]["detail"]
          and "INCAP_DECIDED_BY" in vc[0]["detail"] and "test-report --arm S0X --exam mtw_v2" in vc[0]["detail"]
          and "P1 S1C over S0" in vc[0]["detail"] and "Nothing switches" in vc[0]["detail"],
          [c["detail"][:300] for c in vc])
    its = items(w)
    check("  DS1's record: every build, score, long and verdict item done, its spend "
          "settled (sacct; a build by its experiment's report; the sheets on the lab cost no GPU); the data steps idle "
          "again (each proposed afresh when due); no item failed",
          all(v.get("state") == "done" for k, v in its.items() if k.split(":")[0] in ("build", "score", "long",
                                                                                       "verdict", "sheets"))
          and all(v.get("state") == "missing" for k, v in its.items() if k.startswith("data:"))
          and not [k for k, v in its.items() if str(v.get("state")).startswith("failed")]
          and all(v.get("spend") == "settled" for k, v in its.items() if not k.startswith("sheets:")),
          {k: (v.get("state"), v.get("spend")) for k, v in its.items()})
    n0 = len(pro(w))
    w.tick(3)
    check("  nothing of phase 1 is proposed again", len(pro(w)) == n0, seq(w)[n0:])
    spent = sum(float(v.get("settled") or 0) + float(v.get("prior_gpuh") or 0) for v in items(w).values())
    check("  DS1's GPU-h, settled from sacct and the experiments' reports, under the phase reserve's line (174.2)",
          0 < spent < 174.2, spent)


# ------------------------------------------------------------------ storage
def t_storage():
    section("storage (DS.0 item 21): V3F's floor within phase 1's 60 GB, lane DATA's beyond it")
    w = ds1_world("ds1_storage")
    w.quota_rec = {"ok": True, "used_gb": 6700.0, "quota_gb": 7000.0, "free_gb": 300.0}
    w.tick(2)
    check("300 GB free on 7,000 (V3F's floor 242.2 GB, DATA's 410): the phase-1 screen (2 GB, within 60 GB) proposed",
          seq(w)[:1] == [("L30", "1:screen")], seq(w))
    run(w, ticks=3)
    st = json.loads(S.StreamPaths(str(w.lab), "weed").state(NAME).read_text())
    st["ds1"]["phase1_gb"] = 50.0
    state_write(w, st)
    decide(w, 1, {"src_alpha": "accept", K_SOURCE: "accept", "src_beta": "reject"})
    run(w, ticks=4)
    sc = cards(w, "for storage")
    s_ = seq(w)
    check("phase 1 at 50 GB: the pools (30 GB) would pass its 60 GB, so lane DATA's floor applies (410 GB): no pools, "
          "one card naming the floor; MD-S0's draft (1 GB, within) still proposed",
          ("L30", "1:pools") not in s_ and ("L30", "1:md_s0") in s_ and len(sc) == 1
          and "data:1:pools" in sc[0]["title"] and "lane DATA's floor" in sc[0]["detail"], (s_, [c["title"] for c in sc]))
    w.quota_rec = {"ok": True, "used_gb": 6690.0, "quota_gb": 7000.0, "free_gb": 310.0}
    run(w, ticks=2)
    check("  the quota moving (310 GB free), the pools still waiting: no second card", len(cards(w, "for storage")) == 1,
          [c["title"] for c in cards(w, "for storage")])
    w.quota_rec = {"ok": True, "used_gb": 6500.0, "quota_gb": 7000.0, "free_gb": 500.0}
    run(w, ticks=12, until=lambda: ("L30", "1:pools") in seq(w))
    check("  500 GB free: the pools proposed", ("L30", "1:pools") in seq(w), seq(w))
    wu = ds1_world("ds1_storage_unknown")
    wu.quota_rec = {"ok": True, "quota_gb": None, "free_gb": None}
    wu.tick(3)
    uc = cards(wu, "for storage")
    check("  the quota unknown: no DS1 item, one card naming the waiting items and 'the quota is unknown'",
          not pro(wu) and len(uc) == 1 and "the quota is unknown" in uc[0]["detail"]
          and "data:1:screen" in uc[0]["title"], [c["title"] for c in uc])


# ------------------------------------------------------------------ the family cap and the phase reserve
def _spent(w, gpuh, key="build:S0"):
    st = json.loads(S.StreamPaths(str(w.lab), "weed").state(NAME).read_text())
    st.setdefault("ds1", {"items": {}, "cards": {}}).setdefault("items", {})[key] = {
        "state": "done", "estimate": gpuh, "spend": "settled", "settled": gpuh, "lever": "L32"}
    state_write(w, st)


def t_cap():
    section("the family cap (260 GPU-h) and the phase reserve's line (174.2 GPU-h)")
    w = ds1_world("ds1_cap_line")
    w.tick(1)
    _spent(w, 173.9)
    run(w, ticks=3)
    its = items(w)
    cc = cards(w, "not run (family cap)")
    check("173.9 GPU-h spent: the phase-1 screen (1.0) and MD-S0's draft (0.5) would pass 174.2, the phase reserve's "
          "line: each recorded 'not run (family cap)' with one card naming the line, never proposed",
          its.get("data:1:screen", {}).get("state") == S.DS1_NOT_RUN_CAP
          and its.get("data:1:md_s0", {}).get("state") == S.DS1_NOT_RUN_CAP
          and sorted(c["title"] for c in cc) == ["DS1's data:1:md_s0: not run (family cap)",
                                                 "DS1's data:1:screen: not run (family cap)"]
          and all("phase reserve's line" in c["detail"] for c in cc) and not pro(w, "L30"),
          (its.get("data:1:screen"), [c["title"] for c in cc]))
    run(w, ticks=3)
    check("  never retried: no L30 on later ticks, still one card each", not pro(w, "L30")
          and len(cards(w, "not run (family cap)")) == 2)
    w2 = ds1_world("ds1_cap_total")
    w2.ds1["verdicts"]["2"] = {"status": "decided", "reads": [], "substitution": None}
    w2.tick(1)
    _spent(w2, 259.6)
    run(w2, ticks=3)
    c2 = cards(w2, "data:1:screen: not run (family cap)")
    check("259.6 GPU-h spent (verdict v2 decided, so the line no longer applies): the screen's 1.0 would pass 260: "
          "not run (family cap), the card names the cap", items(w2).get("data:1:screen", {}).get("state") ==
          S.DS1_NOT_RUN_CAP and len(c2) == 1 and "past its cap of 260" in c2[0]["detail"],
          [c["detail"][:200] for c in c2])
    w3 = ds1_world("ds1_cap_after_v2")
    w3.ds1["verdicts"]["2"] = {"status": "decided", "reads": [], "substitution": None}
    w3.tick(1)
    _spent(w3, 200.0)
    run(w3, ticks=2)
    check("200 GPU-h spent with verdict v2 decided: the phase-1 screen is past the line but within the cap: proposed",
          seq(w3)[:1] == [("L30", "1:screen")], seq(w3))
    # S2 and S1c-R are cut first: with the long pair's 42.6 GPU-h left, an S1c-R build that the cap holds alone is
    # refused
    w4 = ds1_world("ds1_cap_cut_last")
    d = w4.ds1
    d["verdicts"]["2"] = {"status": "decided", "reads": [], "substitution": None}
    d.update(ra=True, power_final=True, k_not_run=True)
    d["md_s0"].update(drafts=1, draft="draft_r001", power=True, enlarge=False, frozen=True)
    for ph in ("1", "2"):
        d["phases"][ph].update(revisions=1, new_batches=0)
    built = ("S0", "KF", "S1C", "S1U", "S2")
    d["arms"] = {a: {"images": IMAGES[a], "sha256": "x", "over_line": False} for a in built + ("S1CR",)}
    d["scores"].update({a: "complete" for a in ("K0",) + built})
    w4.tick(1)
    st = json.loads(S.StreamPaths(str(w4.lab), "weed").state(NAME).read_text())
    for a in built:
        st.setdefault("ds1", {"items": {}, "cards": {}})["items"]["build:%s" % a] = {
            "state": "done", "spend": "settled", "settled": 0.0, "lever": "L32"}
    state_write(w4, st)
    _spent(w4, 205.0, key="data:1:pools")
    run(w4, ticks=3)
    c4 = cards(w4, "build:S1CR")
    check("205 GPU-h spent, the long pair's three seeds (42.6) still to come: S1c-R's build (13.4) fits the cap alone "
          "but not with the pair: not run (family cap), cut first (DS.11)", items(w4).get("build:S1CR", {}).get(
              "state") == S.DS1_NOT_RUN_CAP and len(c4) == 1 and "cut first" in c4[0]["detail"],
          (items(w4).get("build:S1CR"), seq(w4), [c["detail"][:200] for c in c4]))


# ------------------------------------------------------------------ failures
def t_fail():
    section("a DS1 failure: one card, once more under a new id; after the second, failed; no lane failure")
    w = ds1_world("ds1_fail")
    name = "inc_build_ds1_data_p1_screen"
    run(w, ticks=3, until=lambda: any(j["name"] == name for j in w.squeue), skip=(name,))
    w.job_done(name, state="FAILED", refusal="[inc2.ds1_data] ERROR: refused")
    run(w, ticks=4, until=lambda: any(j["name"] == name for j in w.squeue), skip=(name,))
    once = dict(items(w).get("data:1:screen") or {})
    ids = [e.get("proposal_id") for e in pro(w, "L30")]
    w.job_done(name, state="FAILED", refusal="[inc2.ds1_data] ERROR: refused")
    run(w, ticks=4, skip=(name,))
    st = w.state()
    rec = st["ds1"]["items"]["data:1:screen"]
    fc = [c["title"] for c in cards(w, "DS1's data step")]
    fd = [c for c in cards(w, "DS1's data step 1:screen failed (L30)")]
    check("L30's screen FAILED (sacct): one card, proposed once more under a new id; after its second failure "
          "recorded failed (a card), its id declined, no lane failure, no pause",
          once.get("state") == "running" and len(set(ids)) == 2 and rec.get("state") == "failed"
          and rec.get("fails") == 2 and fc == ["DS1's data step 1:screen failed (L30)",
                                               "DS1's data step 1:screen failed (L30): recorded failed"]
          and not st.get("paused") and not int(w.lane("MAINT").get("fails") or 0) and ids[-1] in (st.get("declined")
                                                                                                   or []),
          (once.get("state"), ids, fc, rec))
    check("  the card's rerun is the login node's sbatch of the same step",
          fd and "sbatch -p GPU-shared run_inc2_build.sh inc2.ds1_data build --phase 1 --step screen" in fd[0]["detail"],
          [c["detail"][-200:] for c in fd])
    check("  DS1's other items go on: MD-S0's draft proposed after the failure", ("L30", "1:md_s0") in seq(w), seq(w))
    # a long-path seed killed at its walltime fails the arm at once
    wl = ds1_world("ds1_fail_walltime")
    wl.ds1.update(ra=True, power_final=True)
    wl.ds1["md_s0"].update(drafts=1, draft="draft_r001", power=True, enlarge=False, frozen=True)
    wl.ds1["phases"]["1"].update(revisions=1, new_batches=0)
    wl.ds1["arms"] = {"S0": {"images": IMAGES["S0"], "sha256": "x", "over_line": False}}
    seeds = ("inc_long_ds1_s0x_s0", "inc_long_ds1_s0x_s1", "inc_long_ds1_s0x_s2")
    run(wl, ticks=12, until=lambda: any(j["name"] in seeds for j in wl.squeue), skip=seeds + ("inc_build_ds1_s0_m640",))
    wl.job_done("inc_long_ds1_s0x_s", state="TIMEOUT")
    run(wl, ticks=3, skip=seeds + ("inc_build_ds1_s0_m640",))
    r = items(wl).get("long:S0X:train") or {}
    check("a long-path seed killed at its walltime (sacct TIMEOUT): the arm's train recorded failed at once, the why "
          "names the walltime (DS.6)", r.get("state") == "failed" and "walltime" in str(r.get("why")), r)


# ------------------------------------------------------------------ an unknown outcome
def t_lost():
    section("an unknown outcome: followed by its job name and DS1's record")
    w = ds1_world("ds1_lost")
    w.lose_reply = "--step"
    name = "inc_build_ds1_data_p1_screen"
    run(w, ticks=4, until=lambda: (w.lane("MAINT").get("item") or {}).get("uncertain"), skip=(name,))
    it = w.lane("MAINT").get("item") or {}
    ok0 = it.get("lever") == "L30" and it.get("status") == "running" and it.get("uncertain") \
        and not w.state().get("paused")
    w.tick(S.BUILD_LOST_SNAPSHOTS + 2)
    it2 = w.lane("MAINT").get("item") or {}
    ok1 = it2.get("lever") == "L30" and any(j["name"] == name for j in w.squeue)
    w.job_done(name)
    data_step(w, 1, "screen")
    run(w, ticks=4, until=lambda: bool(pro(w, "L31")))
    check("an L30 submission whose outcome is unknown: running while %s is queued (also past %d snapshots), then "
          "done once splits/ds1/summary.json's last step is it; the sheets follow" % (name, S.BUILD_LOST_SNAPSHOTS),
          ok0 and ok1 and items(w).get("data:1:screen", {}).get("state") == "missing"
          and int(items(w).get("data:1:screen", {}).get("runs") or 0) == 1 and pro(w, "L31"),
          (ok0, ok1, it.get("status"), it.get("uncertain"), it2.get("lever"), items(w).get("data:1:screen"),
           seq(w)))


# ------------------------------------------------------------------ the pure plan
def _fs(**kw):
    st = REAL_STATUS(W.TMP / "ds1_plan_empty")
    st["timestamp"] = True
    st.update(ra=True, power_final=True)
    st["md_s0"].update(drafts=1, draft="draft_r001", power=True, enlarge=False, frozen=True)
    for ph in ("1", "2"):
        st["phases"][ph].update(revisions=1, new_batches=0)
    for k, v in kw.items():
        st[k] = v
    return {"status": st, "items": {}, "today": "2026-11-01", "phase1_gb": 0.0}


def t_plan():
    section("ds1_plan (pure): the date substitution, an arm over the line, KD/KT, the verdict's wait")
    arms1 = {a: {"images": IMAGES[a], "sha256": "x", "over_line": False} for a in
             ("S0", "KF", "KD", "KT", "S1CR", "S1C", "S1U")}
    fs = _fs(arms=copy.deepcopy(arms1))
    fs["status"]["long"]["pair"] = {"image_epochs": D.PAIR_IMAGE_EPOCHS}
    fs["status"]["long"]["arms"]["S0X"].update(smoke="complete", trained=[0, 1, 2], walltime_h=24.0)
    fs["items"] = {"build:%s" % a: "done" for a in arms1}
    fs["today"] = "2026-12-06"
    out, waits, cand = DS.ds1_plan(CFG, fs)
    trains = [(x["lever"], x.get("arm"), x.get("verb")) for x in cand["train"]]
    check("after 2026-12-05 with S2 not frozen: the substitution applies; S1Ux's smoke on TRAIN, never S2x's; no "
          "phase-2 screen", cand["substituted"] and ("L32X", "S1UX", "smoke") in trains
          and not [t for t in trains if t[1] == "S2X"]
          and not [x for x in cand["maint"] if x.get("item") == "data:2:screen"], (trains, cand["maint"][:4]))
    fs2 = copy.deepcopy(fs)
    fs2["today"] = "2026-11-20"
    fs2["status"]["phases"]["2"]["revisions"] = 0
    _o, _w, c2 = DS.ds1_plan(CFG, fs2)
    t2 = [(x["lever"], x.get("arm"), x.get("verb")) for x in c2["train"]]
    check("  before it: no S1Ux; the phase-2 screen proposed once S1u is frozen",
          not c2["substituted"] and not [t for t in t2 if t[1] == "S1UX"]
          and [x["item"] for x in c2["maint"]][:1] == ["data:2:screen"], ([x["item"] for x in c2["maint"]], t2))
    # verdict v2 under the substitution: its arms are S0x and S1Ux
    fs3 = copy.deepcopy(fs)
    fs3["status"]["scores"].update({a: "complete" for a in ("S0X", "S1UX")})
    fs3["status"]["verdicts"]["1"] = {"status": "decided"}
    fs3["items"]["verdict:1"] = "done"
    _o, _w, c3 = DS.ds1_plan(CFG, fs3)
    check("  verdict v2 under the substitution once S0x and S1Ux are scored (2_substituted's arms)",
          "verdict:2" in [x["item"] for x in c3["maint"]], [x["item"] for x in c3["maint"]])
    # an arm over the loader's line: moved to the long path (L32X smoke, then train), never an L32 build
    fs4 = _fs(arms={"S0": {"images": IMAGES["S0"], "sha256": "x", "over_line": False},
                    "S1C": {"images": IMAGES["S1C"], "sha256": "y", "over_line": True}})
    fs4["status"]["phases"]["1"]["sources"] = {"src_alpha": source(decided=True, decision="accept", ready=True,
                                                                   pooled=True, md_n=True, mtw_draft=True, mtw=True,
                                                                   checked=True)}
    fs4["status"]["long"]["pair"] = {"image_epochs": D.PAIR_IMAGE_EPOCHS}
    _o, _w, c4 = DS.ds1_plan(CFG, fs4)
    check("an arm over the loader's line (S1c): L32X smoke for it on TRAIN, no L32 build of it",
          ("L32X", "S1C", "smoke") in [(x["lever"], x.get("arm"), x.get("verb")) for x in c4["train"]]
          and "build:S1C" not in [x["item"] for x in c4["maint"]], (c4["train"], [x["item"] for x in c4["maint"]]))
    # KD and KT: only once the K source is pooled; k_not_run settles them
    fs5 = _fs(arms={"S0": {"images": 1, "sha256": "x"}, "KF": {"images": 1, "sha256": "x"}})
    fs5["status"]["phases"]["1"]["sources"] = {K_SOURCE: source(decided=True, decision="accept", ready=True)}
    _o, _w, c5 = DS.ds1_plan(CFG, fs5)
    fs5["status"]["phases"]["1"]["sources"][K_SOURCE]["pooled"] = True
    _o, _w, c6 = DS.ds1_plan(CFG, fs5)
    fs5["status"]["k_not_run"] = True
    _o, _w, c7 = DS.ds1_plan(CFG, fs5)
    check("KD/KT's manifests wait for the K source's pool, then are due; k_not_run.json settles them",
          "data:1:manifests" not in [x["item"] for x in c5["maint"]]
          and "data:1:manifests" in [x["item"] for x in c6["maint"]]
          and "data:1:manifests" not in [x["item"] for x in c7["maint"]],
          ([x["item"] for x in c5["maint"]], [x["item"] for x in c6["maint"]], [x["item"] for x in c7["maint"]]))
    # the verdict waits for its arms; a failed or capped arm does not hold it
    fs8 = _fs(arms=copy.deepcopy(arms1))
    fs8["status"]["scores"].update({a: "complete" for a in ("K0", "KD", "KT", "KF", "S0", "S1C", "S1U")})
    fs8["items"] = {"build:%s" % a: "done" for a in arms1}
    _o, w8, c8 = DS.ds1_plan(CFG, fs8)
    fs8["items"]["build:S1CR"] = S.DS1_NOT_RUN_CAP
    _o, _w, c9 = DS.ds1_plan(CFG, fs8)
    check("verdict v1 waits for S1c-R's score (one wait named); with S1c-R 'not run (family cap)' it is due",
          "verdict:1" not in [x["item"] for x in c8["maint"]] and "S1CR" in str(w8.get("verdict:1"))
          and "verdict:1" in [x["item"] for x in c9["maint"]], (w8, [x["item"] for x in c9["maint"]]))
    # training waits for RA, the power pre-flight and MD-S0's freeze, and names them
    fs10 = _fs(arms=copy.deepcopy(arms1), ra=False)
    _o, w10, c10 = DS.ds1_plan(CFG, fs10)
    check("no build or long-path item before RA; the wait names RA", not [x for x in c10["maint"] if x["lever"] ==
                                                                          "L32"] and not c10["train"]
          and "RA" in str(w10.get("training")), (w10, c10))


def t_plan_rev1():
    section("ds1_plan (pure), Revision 1: SIU never drawn, settled sources, the K source out, S2x's walltime, the bytes")
    arms1 = {a: {"images": IMAGES[a], "sha256": "x", "over_line": False} for a in ("S0", "KF")}
    fs = _fs(arms=copy.deepcopy(arms1))
    fs["status"]["phases"]["1"]["sources"] = {
        K_SOURCE: source(decided=True, decision="accept", ready=True, pooled=True, exam_draw=False),
        "src_alpha": source(decided=True, decision="accept", ready=True, pooled=True, md_n=True, mtw_draft=True,
                            mtw=True, checked=True),
        "src_cap": source(role_c=True, settled_out=True), "src_q": source(quarantined_whole=True, settled_out=True)}
    fs1 = copy.deepcopy(fs)
    fs1["status"]["k_not_run"] = True                # KD and KT settled: the manifests step is due only for S1's
    _o, w1, c1 = DS.ds1_plan(CFG, fs1)
    items1 = [x["item"] for x in c1["maint"]]
    check("SIU accepted and pooled is never drawn (no L30 draw), and phase 1 is settled without it; a role C and a "
          "quarantined source settle with no decision: S1's manifests are due", "data:1:draw" not in items1
          and "data:1:manifests" in items1 and not any(k.startswith("sheets:1") for k in w1), (items1, w1))
    fs2 = copy.deepcopy(fs)
    fs2["status"]["phases"]["1"]["sources"][K_SOURCE].update(decision="reject", settled_out=True, ready=False)
    fs2["status"]["k_source_out"] = True
    arms_all = {a: {"images": IMAGES[a], "sha256": "x", "over_line": False} for a in
                ("S0", "KF", "S1CR", "S1C", "S1U")}
    fs2["status"]["arms"] = arms_all
    fs2["status"]["scores"].update({a: "complete" for a in ("K0", "KF", "S0", "S1CR", "S1C", "S1U")})
    fs2["items"] = {"build:%s" % a: "done" for a in arms_all}
    _o, w2, c2 = DS.ds1_plan(CFG, fs2)
    check("the K source rejected (k_source_out): KD and KT never hold verdict v1, and the manifests step is due once "
          "to record them not run", "verdict:1" in [x["item"] for x in c2["maint"]] and "data:1:manifests" in
          [x["item"] for x in c2["maint"]], ([x["item"] for x in c2["maint"]], w2))
    fs2["status"]["k_not_run"] = True
    _o, _w, c2b = DS.ds1_plan(CFG, fs2)
    check("  once k_not_run.json is written the manifests step is not proposed again",
          "data:1:manifests" not in [x["item"] for x in c2b["maint"]], [x["item"] for x in c2b["maint"]])
    fs3 = _fs(arms={a: {"images": IMAGES[a], "sha256": "x", "over_line": False} for a in ("S0", "S1U", "S2")})
    fs3["status"]["long"]["pair"] = {"image_epochs": D.PAIR_IMAGE_EPOCHS}
    fs3["status"]["long"]["arms"]["S2X"].update(smoke="not run (walltime)")
    fs3["today"] = "2026-11-20"
    _o, _w, c3 = DS.ds1_plan(CFG, fs3)
    check("S2x recorded not run (walltime) before the date: the substitution applies, S1ux's smoke on TRAIN (DS.6)",
          c3["substituted"] and ("L32X", "S1UX", "smoke") in [(x["lever"], x.get("arm"), x.get("verb"))
                                                              for x in c3["train"]], c3["train"])
    fs4 = _fs()
    fs4["status"]["sheets"]["audit_pending"] = ["src_b", "src_a"]
    _o, _w, c4 = DS.ds1_plan(CFG, fs4)
    au = [x for x in c4["maint"] if x["item"].startswith("sheets:audit")]
    check("the audit batch's item names its sources (the cluster's audit set, passed to the lab's render)",
          au and au[0].get("sources") == ["src_a", "src_b"], au)


def t_review():
    section("the ticker, Revision 1's fixes: a DS1 build's blocked runs (a card, no pause, the arm failed), phase 1's "
            "bytes counted once, the audit render's sources and a single render of its set (Revision 2), a second screen "
            "when nothing is new")
    w = ds1_world("ds1_blocked")
    d = w.ds1
    d.update(ra=True, power_final=True, k_not_run=True)
    d["md_s0"].update(drafts=1, draft="draft_r001", power=True, enlarge=False, frozen=True)
    d["phases"]["1"].update(revisions=1, new_batches=0)
    built = ("S0", "KF", "S1CR", "S1C", "S1U")
    d["arms"] = {a: {"images": IMAGES[a], "sha256": "x", "over_line": False} for a in built}
    d["scores"].update({a: "complete" for a in ("K0", "KF", "S1CR", "S1C", "S1U")})
    w.tick(1)
    st = json.loads(S.StreamPaths(str(w.lab), "weed").state(NAME).read_text())
    for a in built:
        st.setdefault("ds1", {"items": {}, "cards": {}})["items"]["build:%s" % a] = {
            "state": "done", "spend": "settled", "settled": 0.0, "lever": "L32", "exp": CFG["build_exps"][a]}
        st["exps"] = list(st.get("exps") or []) + [CFG["build_exps"][a]]      # as L32's start records it
    state_write(w, st)
    for a in built:
        exp = CFG["build_exps"][a]
        if a == "S0":
            w.experiment(exp, done=False, blocked={"base:s1": {"cause": {"kind": "failed_run"},
                                                               "error": "2 failed attempts (CUDA out of memory)"}})
        else:
            w.experiment(exp, final=[w.final_row("base", 0.85, 0.002, 3)])
    run(w, ticks=14, until=lambda: bool(pro(w, "L34")))
    stt = w.state()
    v = pro(w, "L34")
    check("S0's runs blocked (D5, not transient): no pause, one DS1 card, the arm failed for DS1 (no other arm held): "
          "verdict v1 proposed naming --ticker-failed build:S0",
          not stt.get("paused") and w.config().get("enabled") is True and cards(w, "DS1's S0: its runs are blocked")
          and v and "build:S0" in (v[0].get("argv") or [])[(v[0].get("argv") or []).index("--ticker-failed") + 1]
          if v and "--ticker-failed" in (v[0].get("argv") or []) else False,
          (stt.get("paused"), [c["title"] for c in cards(w)][-4:], [e.get("argv") for e in v]))
    # phase 1's bytes: a data step that runs again is counted once
    w2 = ds1_world("ds1_bytes")
    run(w2, ticks=6, until=lambda: w2.ds1["phases"]["1"]["revisions"] == 1)
    w2.ds1["phases"]["1"]["new_batches"] = 2
    run(w2, ticks=6, until=lambda: w2.ds1["phases"]["1"]["revisions"] == 2)
    run(w2, ticks=2)
    f = (w2.state() or {}).get("ds1") or {}
    n_screens = len([x for x in seq(w2) if x == ("L30", "1:screen")])
    kind_of = {"data:1:screen": "data:screen", "data:1:md_s0": "data:md_s0", "data:1:pools": "data:pools",
               "data:1:draw": "data:draw", "data:1:manifests": "data:manifests"}
    want_gb = sum(float(CFG["storage_gb"].get(kind_of.get(k) or ("score" if k.startswith("score") else "build"), 0))
                  for k in f.get("phase1_charged") or [])
    check("phase 1's bytes: two screens charged once (each item's nominal bytes once, DS.11's bounds per item)",
          n_screens == 2 and (f.get("phase1_charged") or []).count("data:1:screen") == 1 and
          abs(float(f.get("phase1_gb") or 0) - want_gb) < 1e-9, (n_screens, f.get("phase1_gb"),
                                                                  f.get("phase1_charged")))
    # the audit render: the lab process gets the sources
    w3 = ds1_world("ds1_audit_sources")
    data_step(w3, 1, "screen")
    w3.ds1["sheets"]["rendered"] = ["source_001", "ln_001", "mtw_001"]
    w3.ds1["sheets"]["audit_pending"] = ["src_beta", "src_alpha"]
    run(w3, ticks=6, until=lambda: any(l["job"].startswith("ds1sheets_audit") for l in w3.runner.launched))
    la = [l["argv"] for l in w3.runner.launched if l["job"].startswith("ds1sheets_audit")]
    check("the owner's audit batch rendered on the lab with --sources (the cluster's audit set)",
          la and la[0][la[0].index("--sources") + 1] == "src_alpha,src_beta", la)
    # Revision 2, item 3: the render ends; the ticker records the set from its own proposal and never renders it again
    run(w3, ticks=8)
    la2 = [l["job"] for l in w3.runner.launched if l["job"].startswith("ds1sheets_audit")]
    f3 = (w3.state() or {}).get("ds1") or {}
    check("  rendered once: the set the render drew is recorded, and no second render of it follows (8 more ticks)",
          len(la2) == 1 and f3.get("audit_render_for") == ["src_alpha", "src_beta"] and
          "audit_001" in w3.ds1["sheets"]["rendered"], (la2, f3.get("audit_render_for")))


def main():
    want = [x for x in os.environ.get("DS1_WORLD", "").split(",") if x]
    try:
        for fn in (t_static, t_plan, t_plan_rev1, t_gate, t_path, t_storage, t_cap, t_fail, t_lost, t_review):
            if want and fn.__name__[2:] not in want:
                continue
            try:
                fn()
            except Exception as e:               # noqa: BLE001
                check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e,
                                                                     traceback.format_exc()[-2500:]))
    finally:
        shutil.rmtree(str(W.TMP), ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
