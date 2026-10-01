#!/usr/bin/env python3
"""A source review (D20's R3 item for a source that failed a pre-check) takes
the source's placement, as D20's own L16 -> L16L does.

L16R always rendered the cluster's fetch (sbatch run_inc_collect.sh fetch).
For a provider placement.json does not place on compute nodes (the candidate's
placement 'lab': the mediaTUM FTP, github), the cluster's collector refuses it
("not_placed_on_cluster: provider mediatum is not placed on compute nodes
(placement.json); the lab hook fetches it", job 47302914), so a person's
approval of such a review could never succeed, and each failure counted
toward the DATA lane's stop-loss. The review of a lab-placed source is now
L16RL (inc_stream_collect_review_lab, R3, a lab action): once a person
approves it, the lab fetch hook runs it detached, and its result folds as an
L16L run's.

Pinned, in the stream world of test_stream_ap_world:
  * the menu, the executor, the policy table and the ticker's lab hooks agree
    on L16RL (R3, a lab action, the fetch hook, priced at zero, never in the
    envelope, never a gated R2 action);
  * a lab-placed candidate failing its pre-check files L16RL for a person, and
    nothing runs before the person decides;
  * approved: the next tick launches the lab fetch (detached) under the
    approval, never an sbatch; the run folds as L16L's (running -> fetched,
    then the sync L16S; or failed, counted against the source);
  * a cluster-placed candidate still files L16R, run by sbatch once approved;
  * an approved L16R (the cluster form) of a source now placed on the lab is
    never submitted: the refusal is recorded with a data card, nothing is
    charged or counted, the approval is closed as not submitted (so a
    person's Run now from the INC page cannot send it to sbatch either), and
    the source's review is filed again in its lab form;
  * the same when the source has no candidate row this tick (fail closed: its
    placement cannot be read);
  * an L16R that already ran in its cluster form and failed for a lab-placed
    source is filed again in its lab form;
  * an L16RL folds as an L16L run in every respect: a partial fetch, the L16
    limits (its run counted for the source: a later fetch's bytes), the
    family in flight, and a person's denial recorded on the source's review.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_world as W  # noqa: E402
from test_stream_ap_world import AP, LS, OWNER, POL, S, X, World, check  # noqa: E402

LAB_SRC = "mediatum_1717366"
CLU_SRC = "hf_unknown"
# licence unknown: D20's pre-check sends the source to a person (L16R / L16RL)
LAB_CAND = {"id": LAB_SRC, "provider": "mediatum", "licence": "unknown", "target_classes": ["Purslane"],
            "bytes": 2e9, "expected_target_boxes": 900}
CLU_CAND = {"id": CLU_SRC, "provider": "hf", "licence": "unknown", "target_classes": ["Purslane"],
            "bytes": 1e9, "expected_target_boxes": 900}


def review_world(tag, cand, placement):
    """A world past R0 with an empty queue, one candidate failing its
    pre-check, and a discovery that just ran (so D20 proposes no L15 and the
    DATA lane stays free): after two ticks D20 has filed the review."""
    w = World(tag)
    w.ready_r0()
    w.queue(0)
    w.placement(placement)
    w.candidates([dict(cand)])
    w.tick()                                      # the first snapshot
    st = w.state()
    st["discover"] = {"last_utc": W.utc(w.clock()), "runs": 1, "found_new": 1, "empty_runs": 0}
    S.StreamPaths(str(w.lab), w.domain).state(W.NAME).write_text(json.dumps(st))
    w.tick()
    return w


def review(w, src):
    return ((w.state() or {}).get("source_reviews") or {}).get(src) or {}


def lab_fetches(w, src):
    return [x for x in w.runner.launched if "fetch" in x["argv"] and src in x["argv"]]


def cluster_fetches(w, src):
    subs = [s for s in w.submits if "run_inc_collect.sh" in " ".join(s["argv"]) and src in s["argv"]]
    verbs = [v for v in w.verbs if v and v[0] == "stream-submit" and src in v]
    return subs + verbs


def run_now(w, aid):
    """What the INC page's Run now does: execute_approved as the person, with
    a Context that can reach the cluster (inc_dashboard._run_action)."""
    cfg = w.config()
    camp = {"name": W.NAME, "autonomy": cfg.get("autonomy") or "off",
            "autonomy_granted_by": cfg.get("autonomy_granted_by"), "envelope_su": cfg.get("envelope_su"),
            "daily_cap_su": cfg.get("daily_cap_su"), "paused_reason": None}
    page = X.Context(slurm_sh=w, lab_repo=str(w.lab), resources=W.RES, clock=w.clock, domain=w.domain)
    return X.execute_approved(aid, camp, page, invoked_by=OWNER)


def stream_camp(w):
    """The campaign as the ticker hands it to the executor (StreamRun.camp),
    read from the state the last tick wrote."""
    sr = S.StreamRun.__new__(S.StreamRun)
    sr.name, sr.cfg, sr.st, sr.dom = W.NAME, w.config(), w.state(), w.dom
    sr._artifact = lambda rel: None
    sr._limit_counts = lambda: {}
    return sr.camp


def approve(w, aid, why="test: a person approves the review"):
    res = AP.decide(w.domain, aid, "approve", OWNER, why, w.clock(), root=str(w.lab))
    assert isinstance(res, dict) and res.get("ok"), ("the person's approval was not recorded", res)


def t_agree():
    print("the menu, the executor, the policy table and the lab hooks agree on L16RL")
    r = LS.row("L16RL")
    act = r["policy_action"]
    desc = POL.describe(act)
    check("L16RL: a sub-lever of L16 on the DATA lane, followed as a lab process, R3",
          LS.family("L16RL") == "L16" and r.get("lane") == "DATA" and r.get("follow") == "lab"
          and r.get("risk") == "R3" and act == "inc_stream_collect_review_lab", r)
    check("  its policy row exists, R3, autopilot and person only, no allocation",
          desc.get("known") and desc.get("risk") == "R3"
          and set(desc.get("allowed_tiers") or []) == {"round-scheduler", "human"}
          and (desc.get("est_su") or {}).get("fixed_su") == 0.0, desc)
    check("  it is one of the L16 family's actions", act in LS.actions_of("L16"), LS.actions_of("L16"))
    check("  a lab action of the executor, never a gated R2 action, an envelope action or a cluster verb",
          act in X.LAB_ACTIONS and act not in X.GATED_R2_ACTIONS and act not in AP.ENVELOPE_ACTIONS
          and act not in X.STREAM_REMOTE and "L16RL" not in LS.envelope_levers()
          and not any(act in v for v in X.STREAM_ENVELOPE_LEVERS.values()))
    params = {"source": LAB_SRC, "max_bytes": 2000000000, "out": "/lab/x/intake/staging/"}
    sr = S.StreamRun.__new__(S.StreamRun)              # the ticker's hooks, without a tick
    sr.runner, sr.utc = W.FakeRunner(), W.utc(W.T0)
    sr.paths = S.StreamPaths(str(W.TMP / "hooks_lab"), "weed")
    hooks = sr._lab_hooks()
    got = hooks[act](dict(params)) if callable(hooks.get(act)) else {}
    ref = hooks["inc_stream_collect_lab"](dict(params))
    check("  the ticker registers a lab hook for it: L16L's fetch hook, launched detached with the same command",
          got.get("ok") and got.get("detached") and got.get("argv") == ref.get("argv")
          and ref["argv"][ref["argv"].index("weed_optimizer_framework.tools.collect") + 1] == "fetch"
          and [x["argv"] for x in sr.runner.launched] == [got.get("argv"), ref["argv"]], (sorted(hooks), got))
    lab_levers = [l for l in LS.lever_ids() if LS.row(l).get("follow") == "lab"]
    check("every lab-followed stream lever's action is a lab action with a hook, and every hook a lab action",
          all(LS.row(l)["policy_action"] in X.LAB_ACTIONS and LS.row(l)["policy_action"] in hooks
              for l in lab_levers) and all(a in X.LAB_ACTIONS for a in hooks), (lab_levers, sorted(hooks)))
    pp = LS.policy_params("L16RL", params)
    argv = LS.render("L16RL", pp)
    rend = X.render(act, pp)
    ok, bad = LS.check_params("L16RL", pp)
    check("  its argv is L16L's command (the collector's fetch on the lab), within the policy bounds, "
          "rendered local and read back by the executor",
          argv == LS.render("L16L", LS.policy_params("L16L", params)) and "sbatch" not in argv and ok
          and rend["local"] and X.argv_check(rend, argv)[0] and X.params_from_argv(act, argv) == pp,
          (argv, bad, rend))
    dom = LS.load_domain("weed")
    est, _d = LS.price("L16RL", params, dom)
    su = POL.estimate_su(act, dict(pp, est_gpu_hours=est))
    check("  priced at zero by the menu and by the policy row (a lab download, no allocation)",
          est == 0.0 and su.get("su") == 0.0, (est, su))


def t_lab_review(outcome):
    print("a lab-placed source failing its pre-check: L16RL, run on the lab once approved (%s)" % outcome)
    w = review_world("rv_lab_%s" % outcome, LAB_CAND, {"hf": "pass", "mediatum": "fail"})
    rv = review(w, LAB_SRC)
    aid = rv.get("approval_id")
    item = w.approvals().get(aid) or {}
    cx = item.get("context") or {}
    check("the review is filed in its lab form (L16RL), R3, pending a person",
          rv.get("lever") == "L16RL" and rv.get("status") == "filed" and item.get("action")
          == "inc_stream_collect_review_lab" and item.get("risk") == "R3" and item.get("status") == "pending",
          (rv, item.get("action"), item.get("risk"), item.get("status")))
    check("  its command is the lab collector's fetch into the lab's staging, no sbatch",
          cx.get("argv") and "sbatch" not in cx["argv"] and "run_inc_collect.sh" not in cx["argv"]
          and "weed_optimizer_framework.tools.collect" in cx["argv"]
          and str(cx["argv"][cx["argv"].index("--out") + 1]).endswith("/intake/staging/"), cx.get("argv"))
    check("  the ledger records the review filed as L16RL",
          [e for e in w.events("review_filed") if e.get("source") == LAB_SRC and e.get("lever") == "L16RL"])
    check("  nothing runs before the person decides (no lab launch, no cluster fetch)",
          not lab_fetches(w, LAB_SRC) and not cluster_fetches(w, LAB_SRC),
          ([x["argv"] for x in w.runner.launched], cluster_fetches(w, LAB_SRC)))
    approve(w, aid)
    w.tick()
    st = w.state()
    it = (st["lanes"]["DATA"].get("item") or {})
    lf = lab_fetches(w, LAB_SRC)
    lab_inc = str(S.StreamPaths(str(w.lab), w.domain).lab_inc)
    check("approved: the next tick launches the fetch on the lab (detached), under the lab's INC tree",
          len(lf) == 1 and lf[0]["argv"][0] == sys.executable and "--inc-dir" in lf[0]["argv"]
          and lf[0]["argv"][lf[0]["argv"].index("--inc-dir") + 1] == lab_inc
          and it.get("lever") == "L16RL" and it.get("status") == "running" and it.get("lab_job"),
          ([x["argv"] for x in lf], it.get("lever"), it.get("status")))
    check("  never an sbatch: no fetch of the source reached the cluster", not cluster_fetches(w, LAB_SRC),
          cluster_fetches(w, LAB_SRC))
    ex = [e for e in w.events("executed") if e.get("lever") == "L16RL"]
    a = w.approvals().get(aid) or {}
    check("  run under the person's approval: the ledger and the approval record it, authorised as the person",
          ex and ex[-1].get("approval_id") == aid and ex[-1].get("decided_by") == OWNER
          and (a.get("execution") or {}).get("phase") == "done"
          and ((a.get("execution") or {}).get("outcome") or {}).get("authorized_as") == OWNER,
          (ex[-1:], a.get("execution")))
    src = st["sources"].get(LAB_SRC) or {}
    check("  its attempt is counted and the source is fetching on the lab, as an L16L run",
          src.get("attempts") == 1 and src.get("status") == "fetching" and src.get("placement") == "lab", src)
    inf = stream_camp(w).get("in_flight")
    check("  while it runs, the executor's campaign counts it in flight as the L16 family (the in_flight limit)",
          inf == {"L16": 1}, inf)
    w.tick()
    check("  launched once, not again on the next tick", len(lab_fetches(w, LAB_SRC)) == 1,
          [x["argv"] for x in w.runner.launched])
    if outcome in ("ok", "partial"):
        w.runner.finish(ok=True, rc=0, tail='[collect] fetch: {"complete": %s, "remaining": %d}'
                        % (("true", 0) if outcome == "ok" else ("false", 7)))
        w.tick()
        st = w.state()
        nxt = (st["lanes"]["DATA"].get("item") or {}).get("lever")
        check("the lab run ends ok: folded as L16L's -- the source is fetched and the sync (L16S) comes next",
              (st["sources"].get(LAB_SRC) or {}).get("status") == "fetched" and nxt == "L16S"
              and [e for e in w.events("item_done") if e.get("lever") == "L16RL"],
              ((st["sources"].get(LAB_SRC)), nxt))
        src = st["sources"].get(LAB_SRC) or {}
        if outcome == "ok":
            check("  its completeness is read from the lab process's last line: complete",
                  src.get("partial") is False and not [e for e in w.events("source_partial")
                                                       if e.get("source") == LAB_SRC], src)
        else:
            check("  its completeness is read from the lab process's last line: partial (shards remain)",
                  src.get("partial") is True and [e for e in w.events("source_partial") if e.get("source") == LAB_SRC],
                  (src, w.events("source_partial")[-1:]))
        # the L16 limits count the review's run as an L16L's: its bytes and its attempt for the source
        want = int(50e9 - 2e9 + 0.5e9)            # 48.5 GB: over 50 GB only with the review's 2 GB counted
        why = X.stream_limits(w.xctx, stream_camp(w), "L16", {"action": "inc_stream_collect_lab", "params": {
            "source": LAB_SRC, "max_bytes": want, "out": "/lab/x/intake/staging/"}})
        mine = [r for r in w.executions() if r.get("action") == "inc_stream_collect_review_lab"
                and r.get("status") == "executed" and (r.get("params") or {}).get("source") == LAB_SRC]
        check("  the L16 limits count its run for the source: a later 48.5 GB fetch of it would reach 50.5 GB",
              len(mine) == 1 and any("source %s would reach 50.5 GB" % LAB_SRC in x for x in why), (why, len(mine)))
    else:
        w.runner.finish(ok=False, rc=1, stderr_tail="the FTP refused the login")
        w.tick()
        st = w.state()
        src = st["sources"].get(LAB_SRC) or {}
        fl = [e for e in w.events("failed") if e.get("lever") == "L16RL"]
        check("the lab run fails: folded as L16L's -- a failed step of the lane, counted against the source",
              fl and src.get("failures") == 1 and src.get("status") == "candidate"
              and int(st["lanes"]["DATA"].get("fails") or 0) == 1
              and (st["lanes"]["DATA"].get("item") or {}).get("lever") != "L16RL", (fl[-1:], src))
    check("  and nothing of the source was ever submitted to the cluster", not cluster_fetches(w, LAB_SRC),
          cluster_fetches(w, LAB_SRC))


def t_cluster_review():
    print("a cluster-placed source failing its pre-check: L16R, unchanged (sbatch once approved)")
    w = review_world("rv_cluster", CLU_CAND, {"hf": "pass"})
    rv = review(w, CLU_SRC)
    aid = rv.get("approval_id")
    item = w.approvals().get(aid) or {}
    argv = (item.get("context") or {}).get("argv") or []
    check("the review is filed in its cluster form (L16R): sbatch run_inc_collect.sh fetch, R3, pending",
          rv.get("lever") == "L16R" and item.get("action") == "inc_stream_collect_review"
          and item.get("risk") == "R3" and item.get("status") == "pending"
          and argv[:5] == ["sbatch", "-p", "GPU-shared", "run_inc_collect.sh", "fetch"], (rv, item.get("action"), argv))
    approve(w, aid)
    w.tick(2)
    a = w.approvals().get(aid) or {}
    check("approved: it is submitted to the cluster through its approval, and nothing is launched on the lab",
          cluster_fetches(w, CLU_SRC) and (a.get("execution") or {}).get("phase") == "done"
          and not lab_fetches(w, CLU_SRC), (cluster_fetches(w, CLU_SRC), a.get("execution")))


def t_legacy_cluster_review():
    print("an approved L16R (the cluster form) of a source placed on the lab is not submitted")
    # filed while its provider read as placed on compute nodes: the cluster form,
    # as every review was before the lab form existed (the live mediatum one)
    w = review_world("rv_legacy", LAB_CAND, {"hf": "pass", "mediatum": "pass"})
    rv = review(w, LAB_SRC)
    aid = rv.get("approval_id")
    pid = ((w.approvals().get(aid) or {}).get("context") or {}).get("proposal_id")
    assert (w.approvals().get(aid) or {}).get("action") == "inc_stream_collect_review", \
        ("set-up: the review is not the cluster form", rv)
    w.placement({"hf": "pass", "mediatum": "fail"})          # placement.json: the provider is the lab's
    w.tick(2)                                                # a snapshot folds it
    approve(w, aid, "test: a person approves the cluster-form review")
    w.tick()
    st = w.state()
    ref = [e for e in w.events("refused") if e.get("approval_id") == aid]
    check_not_submitted(w, aid, "placed on the lab")
    check("  the ledger records why: L16R refused, its source placed on the lab (not_placed_on_cluster)",
          ref and ref[-1].get("lever") == "L16R" and ref[-1].get("source") == LAB_SRC
          and "not_placed_on_cluster" in " ".join(ref[-1].get("reasons") or []), ref[-1:])
    src = st["sources"].get(LAB_SRC) or {}
    check("  nothing is charged or counted: no attempt, no failure, no failed step of the lane, the lane free",
          not src.get("attempts") and not src.get("failures") and not int(st["lanes"]["DATA"].get("fails") or 0)
          and not [e for e in w.events("failed") if e.get("lever") == "L16R"]
          and (st["lanes"]["DATA"].get("item") or {}).get("approval_id") != aid
          and pid in (st.get("declined") or []), (src, st["lanes"]["DATA"]))
    check_run_now_refused(w, aid)
    w.tick(2)
    rv2 = review(w, LAB_SRC)
    item2 = w.approvals().get(rv2.get("approval_id")) or {}
    check("the source's review is filed again in its lab form (L16RL) for a person, the old one recorded",
          rv2.get("lever") == "L16RL" and rv2.get("approval_id") != aid
          and item2.get("action") == "inc_stream_collect_review_lab" and item2.get("status") == "pending"
          and [x.get("approval_id") for x in rv2.get("superseded") or []] == [aid], rv2)
    check("  and the cluster form stays unsubmitted on later ticks; nothing is launched before the person decides",
          not cluster_fetches(w, LAB_SRC) and not lab_fetches(w, LAB_SRC)
          and ((w.approvals().get(aid) or {}).get("execution") or {}).get("phase") == "failed",
          (cluster_fetches(w, LAB_SRC), [x["argv"] for x in w.runner.launched]))


def check_not_submitted(w, aid, why_part):
    """An approved L16R the lane refused: never submitted; its approval closed
    as not submitted in the approvals log only (no job, no execution-log
    record, no charge); a data card names the approval."""
    a = w.approvals().get(aid) or {}
    ex = a.get("execution") or {}
    oc = ex.get("outcome") or {}
    check("the approved cluster fetch is never submitted; its approval is closed as not submitted (no job), "
          "so it no longer awaits execution",
          not cluster_fetches(w, LAB_SRC) and ex.get("phase") == "failed" and oc.get("status") == "not_submitted"
          and oc.get("job_ids") == [] and why_part in str(oc.get("error"))
          and aid not in [x.get("id") for x in AP.awaiting_execution(w.domain, root=str(w.lab))],
          (cluster_fetches(w, LAB_SRC), ex))
    ran = [r for r in w.executions() if r.get("approval_id") == aid and (r.get("status") == "executed"
                                                                         or r.get("charged"))]
    check("  the execution log and the budget are untouched: no run or charge of the approval is recorded there",
          not ran, ran[-1:])
    st = w.state()
    cards = [c for c in st.get("cards") or [] if c.get("kind") == "data" and c.get("lever") == "L16R"
             and aid in str(c.get("title")) and LAB_SRC in str(c.get("title"))]
    check("  a data card names the source and the approval, and says the approval is closed",
          cards and "closed" in str(cards[-1].get("detail")) and why_part in str(cards[-1].get("detail")),
          [c.get("title") for c in st.get("cards") or []])


def check_run_now_refused(w, aid):
    res = run_now(w, aid)
    check("  a person's Run now from the INC page is refused (the approval is closed): no sbatch reaches the cluster",
          res.get("status") == "refused" and "already executed" in " ".join(res.get("reasons") or [])
          and not cluster_fetches(w, LAB_SRC), (res.get("status"), res.get("reasons"), cluster_fetches(w, LAB_SRC)))


def t_absent_candidate():
    print("an approved L16R whose source has no candidate row this tick is not submitted (fail closed)")
    w = review_world("rv_absent", LAB_CAND, {"hf": "pass", "mediatum": "pass"})
    rv = review(w, LAB_SRC)
    aid = rv.get("approval_id")
    assert (w.approvals().get(aid) or {}).get("action") == "inc_stream_collect_review", \
        ("set-up: the review is not the cluster form", rv)
    w.placement({"hf": "pass", "mediatum": "fail"})          # the provider is the lab's
    w.candidates([])                                          # and the next plan (L15) no longer lists the source
    w.tick(2)
    approve(w, aid, "test: a person approves the cluster-form review")
    w.tick()
    st = w.state()
    ref = [e for e in w.events("refused") if e.get("approval_id") == aid]
    check_not_submitted(w, aid, "no candidate row")
    check("  the ledger records why: no candidate row, so its placement cannot be read",
          ref and ref[-1].get("lever") == "L16R" and "no candidate row" in " ".join(ref[-1].get("reasons") or []),
          ref[-1:])
    src = st["sources"].get(LAB_SRC) or {}
    check("  nothing is counted, the lane is free, and the source's review is superseded",
          not src.get("attempts") and not src.get("failures") and not int(st["lanes"]["DATA"].get("fails") or 0)
          and (st["lanes"]["DATA"].get("item") or {}).get("approval_id") != aid
          and review(w, LAB_SRC).get("status") == "superseded", (src, review(w, LAB_SRC)))
    check_run_now_refused(w, aid)


def t_ran_cluster_review():
    print("an L16R that already ran in its cluster form for a lab-placed source: its review is filed again as L16RL")
    # the live record: approval ap-1790700040-42144765 ran as job 47302914 before the lab form existed
    w = review_world("rv_ran", LAB_CAND, {"hf": "pass", "mediatum": "pass"})
    aid = review(w, LAB_SRC).get("approval_id")
    approve(w, aid, "test: a person approves the cluster-form review")
    w.tick()
    a = w.approvals().get(aid) or {}
    sent = len(cluster_fetches(w, LAB_SRC))
    assert sent and (a.get("execution") or {}).get("phase") == "done", \
        ("set-up: the cluster-form review was not submitted", a.get("execution"))
    w.placement({"hf": "pass", "mediatum": "fail"})          # placement.json: the provider is the lab's
    w.job_done("inc_stream_collect_", state="FAILED",
               refusal="not_placed_on_cluster: provider mediatum is not placed on compute nodes (placement.json); "
                       "the lab hook fetches it")
    w.tick(3)
    st = w.state()
    src = st["sources"].get(LAB_SRC) or {}
    rv2 = review(w, LAB_SRC)
    item2 = w.approvals().get(rv2.get("approval_id")) or {}
    check("the failed cluster run is folded once (one attempt, one failure)",
          src.get("attempts") == 1 and src.get("failures") == 1 and src.get("status") == "candidate", src)
    check("  the source's review is filed again in its lab form (L16RL) for a person, the spent one recorded",
          rv2.get("lever") == "L16RL" and rv2.get("approval_id") != aid
          and item2.get("action") == "inc_stream_collect_review_lab" and item2.get("status") == "pending"
          and [x.get("approval_id") for x in rv2.get("superseded") or []] == [aid]
          and "already ran" in str((rv2.get("superseded") or [{}])[-1].get("reason")), rv2)
    check("  nothing else is submitted or launched before the person decides",
          len(cluster_fetches(w, LAB_SRC)) == sent and not lab_fetches(w, LAB_SRC),
          (cluster_fetches(w, LAB_SRC), [x["argv"] for x in w.runner.launched]))
    w.tick(2)
    check("  and it is filed once: no further review of the source on later ticks",
          review(w, LAB_SRC).get("approval_id") == rv2.get("approval_id")
          and len([e for e in w.events("review_filed") if e.get("source") == LAB_SRC]) == 2,
          [e.get("lever") for e in w.events("review_filed") if e.get("source") == LAB_SRC])


def t_deny_lab_review():
    print("a person denies an L16RL: recorded on the source's review, nothing runs")
    w = review_world("rv_deny", LAB_CAND, {"hf": "pass", "mediatum": "fail"})
    aid = review(w, LAB_SRC).get("approval_id")
    res = AP.decide(w.domain, aid, "deny", OWNER, "test: a person denies the review", w.clock(), root=str(w.lab))
    assert res.get("ok"), res
    w.tick(2)
    rv = review(w, LAB_SRC)
    check("the denied L16RL is recorded on source_reviews and declined; it is not filed again",
          rv.get("status") == "denied" and rv.get("lever") == "L16RL" and rv.get("approval_id") == aid
          and [e for e in w.events("denied") if e.get("approval_id") == aid]
          and len([e for e in w.events("review_filed") if e.get("source") == LAB_SRC]) == 1, rv)
    check("  nothing runs: no lab launch, no cluster fetch", not lab_fetches(w, LAB_SRC)
          and not cluster_fetches(w, LAB_SRC), [x["argv"] for x in w.runner.launched])


def main():
    W.run_case("agree", t_agree)
    W.run_case("lab_ok", lambda: t_lab_review("ok"))
    W.run_case("lab_partial", lambda: t_lab_review("partial"))
    W.run_case("lab_failed", lambda: t_lab_review("failed"))
    W.run_case("lab_denied", t_deny_lab_review)
    W.run_case("cluster", t_cluster_review)
    W.run_case("legacy", t_legacy_cluster_review)
    W.run_case("absent", t_absent_candidate)
    W.run_case("ran", t_ran_cluster_review)
    return W.closing()


if __name__ == "__main__":
    sys.exit(main())
