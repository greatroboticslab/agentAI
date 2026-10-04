"""Reports of the continuous loop's stream (docs/CONTINUOUS_LOOP.md §3.7).

    python -m weed_optimizer_framework.tools.inc2.stream_report --stream SID
    python -m weed_optimizer_framework.tools.inc2.stream_report --segment SID_sNNN

Per stream: INC_DIR/stream/<sid>/report.{md,json}. It leads with the success
measure: the sealed cwd12 test mAP50-95 (the 12-class score, species_map50_95,
as inc/report.py reports it) of the latest milestone, mean +- sd over its cold
seeds, and the gap to 0.90; the class-agnostic test score beside it (§11 risk
4: which of the two gaps is closing) and the chain incumbent's test score as
the secondary number. Before any stream milestone the headline is milestone 0
(the R0 baseline of the chosen arm, and every capacity arm of L-4, each read
once). Then:
  * the milestone table: dev, ImageWeeds (same-lab for every v2 model, D-A)
    and test, mean +- sd over the seeds, the chain incumbent, the 5 v 5 dev
    verdict against the previous good milestone, the gap to 0.90 and the
    research_only flag of the models (§8);
  * the timeline of every increment: sources, target boxes per species, the
    pinned gate's verdict and Protocol v3's, P_data, P_recipe, the recorded
    attribution.blame, the truth verdict and the disposition;
  * pool size, M / |P| and dev per segment (the base seeds on P_{s-1} and the
    chosen chain's final incumbent);
  * per-source yield (§7.4): the stream's cuts and dispositions per source,
    and the intake record (INC_DIR/intake/sources.jsonl) where it exists;
  * SU spent against the L-2 envelope (1,000 SU to 2026-12-31, 350 per
    month), from the run seconds of every stream experiment;
  * supply: the eligible queue, holds, holds past their deadline.

Per segment: INC_DIR/<exp>/report.{md,json} in the pinned inc/report.py's
conventions (its ledger reader, step rows, gate record, score reader, run
seconds), with the segment's exams only (dev, imageweeds), plus the stream's
Protocol v3 reading and dispositions once the segment is committed.

At every compared milestone, milestones/mNNN/research_log_entry.md: the
RESEARCH_LOG entry (what changed, why, how it was verified) for whoever
appends it; this module never edits RESEARCH_LOG.md.

Numbers here are read from score and run files; nothing is estimated except
where a line says est.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import sys
from pathlib import Path

from ..inc import common as C
from ..inc import driver as D
from ..inc import report as R
from . import stream as ST

TARGET = 0.90
SEGMENT_EXAMS = ST.SEGMENT_EXAMS
MILESTONE_EXAMS = ST.MILESTONE_EXAMS
SU_PER_GPU_H = 1.0                     # V100 (brain/su_rates.json); H100 would be 2


def log(msg):
    print("[inc2.stream_report] %s" % msg, flush=True)


def _read(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def _f(x, nd=4):
    return "-" if x is None else ("%.*f" % (nd, x))


def _ms(m):
    if not m or m.get("n", 0) == 0 or m.get("mean") is None:
        return "-"
    if m.get("sd") is None:
        return "%.4f" % m["mean"]
    return "%.4f +- %.4f" % (m["mean"], m["sd"])


def _gap(mean):
    return None if mean is None else TARGET - mean


def _write_text(path, text):
    ST._write_text(path, text)


# ------------------------------------------------------------ score reading
def exam_values(exp, rids, exam):
    """{"twelve": mean_sd, "agnostic": mean_sd, "production": [...], "runs": n} of
    the given runs' scores on an exam (inc/report.py score_values)."""
    paths = D.Paths(exp)
    got = [R.score_values(paths, rid, exam) for rid in rids]
    got = [g for g in got if g is not None]
    return {"twelve": R.mean_sd([g[0] for g in got]), "agnostic": R.mean_sd([g[1] for g in got]),
            "production": sorted({g[2] for g in got}, key=str), "runs": len(got)}


def baseline_table(exp, exams=MILESTONE_EXAMS):
    """A baseline experiment's final runs (final__base__sN) on every exam."""
    paths = D.Paths(exp)
    defn = _read(paths.exp_json)
    if not defn:
        return None
    rids = ["final__base__s%d" % s for s in defn["seeds"]]
    out = {"exp": exp, "seeds": defn["seeds"], "n_images": (defn.get("base") or {}).get("n_images"),
           "done": bool((_read(paths.state) or {}).get("done")), "exams": {}}
    for e in exams:
        out["exams"][e] = exam_values(exp, rids, e)
    return out


def incumbent_scores(stream, n):
    """The chain incumbent's secondary scores of milestone n: the run
    inc2.baseline secondary wrote into the milestone experiment
    (runs/secondary__incumbent/scores/<exam>.json); milestones/mNNN/
    incumbent.json says which incumbent it is."""
    rec = _read(stream.p.milestone_dir(n) / "incumbent.json") or {}
    m = stream.fold.milestones.get(n) or {}
    out = {"status": rec.get("status"), "segment": rec.get("segment"), "incumbent": rec.get("incumbent"),
           "exams": {}}
    if m.get("exp"):
        paths = D.Paths(m["exp"])
        for e in MILESTONE_EXAMS:
            s = _read(paths.score(rec.get("run_id") or ST.SECONDARY_RUN, e))
            if s:
                out["exams"][e] = {"twelve": s.get("species_map50_95", s.get("map50_95")),
                                   "agnostic": s.get("agnostic_map50_95"), "production": s.get("production")}
    return out


def gpu_hours(exp):
    """(hours, runs) of every run of an experiment (inc/report.py run_seconds)."""
    paths = D.Paths(exp)
    st = _read(paths.state)
    if not st:
        return 0.0, 0
    tot = 0.0
    for rid, r in st.get("runs", {}).items():
        tot += R.run_seconds(paths, rid, r)
    return tot / 3600.0, len(st.get("runs", {}))


def month_hours(exp, month):
    """GPU-h of an experiment's runs created in the given YYYY-MM."""
    paths = D.Paths(exp)
    st = _read(paths.state) or {}
    tot = 0.0
    for rid, r in st.get("runs", {}).items():
        if str(r.get("created_utc", "")).startswith(month):
            tot += R.run_seconds(paths, rid, r)
    return tot / 3600.0


# -------------------------------------------------------------- segments
def segment_report(exp, stream=None):
    """INC_DIR/<exp>/report.{json,md} for a stream segment (or the Stage C
    chain): the pinned report's step rows, gate record, final table on the
    segment's exams, GPU-hours, plus the stream's v3 reading when committed."""
    paths = D.Paths(exp)
    defn = _read(paths.exp_json)
    state = _read(paths.state)
    if not defn or not state:
        raise ST.StreamError("%s is not an initialised experiment" % exp)
    notes = []
    ledger = R.read_ledger(paths, notes)
    exams = list(defn.get("final_exams") or SEGMENT_EXAMS)
    rep = {"exp": exp, "type": defn["type"], "testing": bool(defn.get("testing")), "generated_utc": D._utc(),
           "done": bool(state.get("done")), "done_utc": state.get("done_utc"), "blocked": state.get("blocked", {}),
           "seeds": defn["seeds"], "exams": exams, "notes": notes, "code": state.get("code"),
           "replay_mode": defn.get("replay_mode", D.DEFAULT_REPLAY_MODE), "gate": R.gate_record(defn, state, ledger),
           "stream": defn.get("stream"), "arm": defn.get("arm"), "protocol": defn.get("protocol"),
           "interventions": [e for e in ledger.values() if e.get("type") in ("unblock", "code_repin")]}
    if defn["type"] == "chain":
        rep["steps"] = R._step_rows(defn, ledger, state)
        rep["chains"] = {r: {"phase": ch["phase"], "accepted": ch["accepted"], "neutral": ch["neutral"],
                             "quarantined": ch["quarantined"],
                             "final_incumbent": (ch.get("incumbent") or {}).get("run_id"),
                             "recipe": defn["recipes"][r]} for r, ch in state["chains"].items()}
    final = []
    for label, ids in R._final_groups(defn):
        row = {"model": label, "runs": ids, "exams": {}}
        for e in exams:
            v = exam_values(exp, ids, e)
            row["exams"][e] = {"twelve": v["twelve"], "agnostic": v["agnostic"]}
        final.append(row)
    rep["final"] = final
    by_owner, n_runs = collections.defaultdict(float), collections.Counter()
    for rid, r in state["runs"].items():
        by_owner[r["owner"]] += R.run_seconds(paths, rid, r)
        n_runs[r["owner"]] += 1
    rep["gpu_hours"] = {o: {"runs": n_runs[o], "hours": by_owner[o] / 3600.0} for o in sorted(by_owner)}
    rep["gpu_hours_total"] = sum(by_owner.values()) / 3600.0
    commit = None
    if stream is not None and stream.fold is not None:
        seg = next((s for s in stream.fold.segments.values() if s["exp"] == exp), None)
        if seg and seg.get("commit"):
            commit = seg["commit"]
    rep["stream_commit"] = commit
    rep["research_only"] = ((defn.get("stream") or {}).get("research_only") or {}).get("models")
    D._write_json(paths.root / "report.json", rep)
    _write_text(paths.root / "report.md", render_segment(rep))
    return rep


def render_segment(rep):
    L = ["# Segment %s%s" % (rep["exp"], " [TESTING]" if rep["testing"] else ""), ""]
    s = rep.get("stream") or {}
    L.append("Stream %s, segment %s, base %s; replay %s; gate %s; truth %s; done %s."
             % (s.get("sid"), s.get("segment"), s.get("pool"), rep["replay_mode"],
                "net flips (v2 record), read under Protocol v3 at commit" if rep.get("gate") else "-",
                "on" if (s.get("truth_policy") or {}).get("on") else "off", "yes" if rep["done"] else "no"))
    L.append("Models trained on research_only data: %s." % rep.get("research_only"))
    L.append("")
    if rep.get("steps"):
        L.append("## Steps (pinned gate record)")
        L.append("")
        L.append("| Step | Chain | Verdict | P_data | P_recipe | Guards (reg/spe/flip) | blame | Truth |")
        L.append("|---|---|---|---|---|---|---|---|")
        for st in rep["steps"]:
            for r, c in st["chains"].items():
                if c is None:
                    L.append("| %s | %s | - | | | | | |" % (st["tag"], r))
                    continue
                g = c["guards"]
                L.append("| %s | %s | %s | %.3f | %.3f | %s/%s/%s | %s | %s |"
                         % (st["tag"], r, c["verdict"], c["p_data"], c["p_recipe"], _yn(g["regression"]),
                            _yn(g["species"]), _yn(g["flips"]), c["attribution"]["blame"],
                            (st["truth"] or {}).get("verdict", "-")))
        L.append("")
    cm = rep.get("stream_commit")
    if cm:
        L.append("## Commit (Protocol v3 reading)")
        L.append("")
        L.append("Chosen chain: %s. D30 %s; D33 species %s." % (cm.get("chosen"), (cm.get("d30") or {}).get("fires"),
                                                            (cm.get("d33") or {}).get("species")))
        L.append("")
        L.append("| Increment | v3 verdict | Species failed (v3) | Disposition |")
        L.append("|---|---|---|---|")
        for x in (cm.get("steps") or {}).get(cm.get("chosen"), []):
            L.append("| %s | %s | %s | %s |" % (x["increment"], x["v3"]["verdict"],
                                               ", ".join(x["v3"]["species_failed"]) or "-", x["disposition"]))
        L.append("")
    L.append("## Final table (segment exams: %s)" % ", ".join(rep["exams"]))
    L.append("")
    head = ["Model"] + ["%s %s" % (e, k) for e in rep["exams"] for k in ("12-class", "agn")]
    L.append("| " + " | ".join(head) + " |")
    L.append("|" + "---|" * len(head))
    for row in rep["final"]:
        cells = [row["model"]]
        for e in rep["exams"]:
            cells += [_ms(row["exams"][e]["twelve"]), _ms(row["exams"][e]["agnostic"])]
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    L.append("GPU-hours: %.2f (%s)." % (rep["gpu_hours_total"], ", ".join(
        "%s %.2f" % (o, v["hours"]) for o, v in rep["gpu_hours"].items())))
    return "\n".join(L) + "\n"


def _yn(b):
    return "-" if b is None else ("y" if b else "n")


# ------------------------------------------------------------- milestones
def milestone_row(stream, n, m):
    tab = baseline_table(m["exp"]) or {"exams": {}}
    ex = tab.get("exams", {})
    test = (ex.get("test") or {}).get("twelve") or {}
    pool = stream.fold.pools.get(m["pool"]) or {}
    row = {"n": n, "exp": m["exp"], "pool": m["pool"], "pool_images": pool.get("n_images"),
           "state": m.get("state"), "verdict": m.get("verdict"), "compared_with": m.get("compared_with"),
           "perm_p": m.get("perm_p"), "exams": ex, "gap_to_target": _gap(test.get("mean")),
           "research_only": pool.get("research_only"), "incumbent": incumbent_scores(stream, n) if n > 0 else None,
           "done": tab.get("done")}
    cmp = _read(stream.p.milestone_dir(n) / "comparison.json") if n > 0 else None
    if cmp:
        row["dev_comparison"] = {k: cmp.get(k) for k in ("new_mean", "new_sd", "old_mean", "old_sd", "perm_p",
                                                         "truth_p", "truth_verdict", "species_failed", "verdict")}
    return row


def headline(rows):
    """The latest milestone with a test score (stream milestones first, then
    milestone 0)."""
    for r in sorted(rows, key=lambda r: -r["n"]):
        t = ((r["exams"].get("test") or {}).get("twelve") or {})
        if t.get("mean") is not None:
            a = ((r["exams"].get("test") or {}).get("agnostic") or {})
            return {"milestone": r["exp"], "n": r["n"], "pool": r["pool"], "pool_images": r["pool_images"],
                    "test_mean": t["mean"], "test_sd": t.get("sd"), "seeds": t.get("n"),
                    "agnostic_mean": a.get("mean"), "agnostic_sd": a.get("sd"), "gap_to_target": _gap(t["mean"]),
                    "incumbent_test": ((r.get("incumbent") or {}).get("exams", {}).get("test") or {}).get("twelve"),
                    "research_only": r.get("research_only")}
    return None


def milestone_entry(stream, n):
    """milestones/mNNN/research_log_entry.md: the RESEARCH_LOG entry of a
    compared milestone (what changed, why, how verified)."""
    f = stream.fold
    m = f.milestones[n]
    row = milestone_row(stream, n, m)
    prev = next((x for x in f.milestones.values() if x["exp"] == m.get("compared_with")), None)
    pool = f.pools.get(m["pool"]) or {}
    prev_pool = f.pools.get((prev or {}).get("pool")) or {}
    lineage = f.pool_lineage(m["pool"])
    added = []
    for name in lineage[:lineage.index(prev["pool"])] if prev and prev["pool"] in lineage else lineage[:-1]:
        added += [x for x in (f.pools[name].get("parts") or [])[1:]]
    added_images = sum(f.increments[i]["n_images"] for i in added if i in f.increments)
    sources = collections.Counter()
    for i in added:
        for s, k in (f.increments.get(i, {}).get("sources") or {}).items():
            sources[s] += k
    t = ((row["exams"].get("test") or {}).get("twelve") or {})
    iw = ((row["exams"].get("imageweeds") or {}).get("twelve") or {})
    dv = ((row["exams"].get("dev") or {}).get("twelve") or {})
    cmp = row.get("dev_comparison") or {}
    day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    L = ["## %s: stream %s milestone %s (%s)" % (day, stream.sid, m["exp"], "TESTING" if f.defn.get("testing") else
                                                "production"), ""]
    L.append("**What changed.** The pool went from %s (%s images) to %s (%s images): %d accepted increment(s) of M = %d "
             "(%d images) from %s." % (prev_pool.get("name", "-"), prev_pool.get("n_images", "-"), pool.get("name"),
                                       pool.get("n_images"), len(added), f.defn["M"], added_images,
                                       ", ".join("%s %d" % kv for kv in sorted(sources.items())) or "no source"))
    L.append("")
    L.append("**Why.** The milestone rule (§5.5): %s. Models trained on research_only data: %s."
             % (", ".join(stream.summary().get("milestones", {}).get("due_reasons") or ["triggered"]),
                pool.get("research_only")))
    L.append("")
    L.append("**How it was verified.** %d cold seeds on the pool, scored by the locked scorer (inc/scorer.py) on dev, "
             "ImageWeeds and the sealed test; the decision used dev only: against %s, mean %s vs %s, one-sided "
             "permutation p = %s (rollback only at p <= %.3f with a lower mean), truth P = %s, species guard failed: %s. "
             "Verdict: %s."
             % (dv.get("n") or 0, m.get("compared_with"), _f(cmp.get("new_mean")), _f(cmp.get("old_mean")),
                _f(cmp.get("perm_p")), ST.PERM_ALPHA, _f(cmp.get("truth_p"), 3),
                ", ".join(cmp.get("species_failed") or []) or "none", m.get("verdict")))
    L.append("")
    L.append("**Result.** Sealed test mAP50-95 %s over %s seeds; gap to 0.90: %s. Class-agnostic test %s. ImageWeeds "
             "(same-lab for every v2 model) %s. Dev %s."
             % (_ms(t), t.get("n") or 0, _f(_gap(t.get("mean"))),
                _ms(((row["exams"].get("test") or {}).get("agnostic") or {})), _ms(iw), _ms(dv)))
    text = "\n".join(L) + "\n"
    d = stream.p.milestone_dir(n)
    _write_text(d / "research_log_entry.md", text)
    return text


# ------------------------------------------------------------------ stream
def build(sid, stream=None):
    """INC_DIR/stream/<sid>/report.{json,md}."""
    st = stream or ST.Stream(sid, quiet=True)
    if st.fold is None:
        st.load()
    f = st.fold
    arms = []
    arm_ev = [e for e in ST.Ledger(st.p.ledger).read() if e["event"] == "arm"]   # never the writer's own fence
    if arm_ev:
        for a, r in (arm_ev[-1].get("arms") or {}).items():
            tab = baseline_table(r["exp"])
            t = (((tab or {}).get("exams") or {}).get("test") or {}).get("twelve") or {}
            arms.append({"arm": a, "exp": r["exp"], "dev_mean": r.get("mean"), "dev_sd": r.get("sd"),
                         "test": t, "gap_to_target": _gap(t.get("mean")),
                         "chosen": a == (f.arm or {}).get("id")})
    ms_rows = [milestone_row(st, n, m) for n, m in f.milestones.items()]
    head = headline(ms_rows)
    if head:
        # after a rollback the latest milestone's pool is no longer the stream's: say so beside the number
        hr = next((r for r in ms_rows if r["exp"] == head["milestone"]), {})
        cur = f.current_pool()["name"]
        head.update(verdict=hr.get("verdict"), current_pool=cur,
                    rolled_back=head.get("pool") not in set(f.pool_lineage()))
    timeline = []
    for inc, rec in f.increments.items():
        seg = f.segments.get(rec["segment"]) or {}
        c = seg.get("commit") or {}
        step = next((x for x in (c.get("steps") or {}).get(c.get("chosen"), []) if x["increment"] == inc), None)
        timeline.append({"increment": inc, "segment": seg.get("exp"), "step": rec["step"], "images": rec["n_images"],
                         "sources": rec.get("sources"),
                         "target_boxes": {k: v for k, v in (rec.get("species_boxes") or {}).items() if v},
                         "pinned_verdict": (step or {}).get("pinned", {}).get("verdict"),
                         "v3_verdict": (step or {}).get("v3", {}).get("verdict"),
                         "p_data": (step or {}).get("v3", {}).get("p_data"),
                         "p_recipe": (step or {}).get("v3", {}).get("p_recipe"),
                         "blame": (step or {}).get("pinned", {}).get("blame"), "truth": (step or {}).get("truth"),
                         "disposition": rec.get("disposition"), "status": rec.get("status")})
    segs = []
    for s in f.segments.values():
        pool = f.pools.get(s["base_pool"]) or {}
        try:
            dev = st._base_dev(s["exp"])
        except ST.StreamError:
            dev = []
        c = s.get("commit") or {}
        inc_dev = None
        if c.get("chosen"):
            v = exam_values(s["exp"], ["final__%s__incumbent" % c["chosen"]], "dev")
            inc_dev = v["twelve"].get("mean")
        segs.append({"exp": s["exp"], "state": s["state"], "base_pool": s["base_pool"], "pool_images": pool.get("n_images"),
                     "m_over_pool": (f.defn["M"] / float(pool["n_images"])) if pool.get("n_images") else None,
                     "k": s["k"], "recipes": s["recipes"], "truth": s["truth"], "base_dev": R.mean_sd(dev),
                     "incumbent_dev": inc_dev, "chosen": c.get("chosen"),
                     "dispositions": c.get("dispositions")})
    yield_ = collections.OrderedDict()
    for inc, rec in f.increments.items():
        for src, k in (rec.get("sources") or {}).items():
            y = yield_.setdefault(src, collections.Counter())
            y["cut"] += k
            y[rec.get("disposition") or rec.get("status") or "in_segment"] += k
            if rec.get("disposition") and rec.get("status") not in (None, rec.get("disposition")):
                y["then_%s" % rec["status"]] += k        # a later rollback (suspect) or bisect decision
    intake = collections.OrderedDict()
    for r in ST._read_jsonl(Path(C.INC_DIR) / "intake" / "sources.jsonl"):
        if isinstance(r, dict) and r.get("source"):
            intake[r["source"]] = {k: r.get(k) for k in ("status", "attempts", "bytes", "su", "yield", "provider",
                                                         "admitted_target_images", "verified_target_boxes")
                                   if k in r}
    exps = [s["exp"] for s in f.segments.values()] + [m["exp"] for n, m in f.milestones.items() if n > 0]
    exps += [e for b in f.bisects.values() for e in b["arms"].values()]
    if f.feasibility and f.feasibility.get("exp"):
        exps.append(f.feasibility["exp"])
    month = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")
    su = {}
    for e in exps:
        h, n = gpu_hours(e)
        su[e] = {"gpu_h": h, "runs": n, "month_gpu_h": month_hours(e, month)}
    spent = sum(v["gpu_h"] for v in su.values()) * SU_PER_GPU_H
    month_spent = sum(v["month_gpu_h"] for v in su.values()) * SU_PER_GPU_H
    summ = st.summary()
    budget = f.defn.get("budget") or ST.BUDGET
    # The monthly window is the current rule's (ST.BUDGET: none since the
    # 2026-10-04 amendment, docs/CONTINUOUS_LOOP.md 6.6), not the stream
    # definition's record: a stream defined before the amendment still records
    # L-2's 350 SU there, which no longer applies. A window a campaign declares
    # lives in its config, which this report does not read.
    window_cap = ST.BUDGET.get("window_cap_su")
    rep = {"format": "inc2-stream-report/1", "sid": st.sid, "generated_utc": D._utc(), "testing": bool(f.defn.get("testing")),
           "target": TARGET, "headline": head, "capacity_arms": arms, "milestones": ms_rows, "timeline": timeline,
           "segments": segs, "yield": {"stream": {k: dict(v) for k, v in yield_.items()}, "intake": intake},
           "su": {"per_experiment": su, "spent_su": spent, "month": month, "month_su": month_spent,
                  "envelope_su": budget.get("envelope_su"), "window_cap_su": window_cap,
                  "until": budget.get("until"), "note": "GPU-hours from run.json seconds (V100: 1 SU per GPU-h); build "
                                                        "and scorer jobs are not included"},
           "supply": summ.get("queue"), "pool": summ.get("pool"), "M": f.defn["M"],
           "quarantined_sources": f.q_sources, "suspect_increments": summ.get("suspect_increments"),
           "feasibility": f.feasibility, "arm": f.arm, "chosen_recipe": f.chosen_recipe}
    D._write_json(st.p.report_json, rep)
    _write_text(st.p.report_md, render(rep))
    return rep


def render(rep):
    L = ["# Stream %s%s" % (rep["sid"], " [TESTING]" if rep["testing"] else ""), ""]
    h = rep["headline"]
    if h:
        L.append("**Sealed test mAP50-95 (cwd12, 12-class): %.4f%s over %s cold seeds at %s** (pool %s, %s images). "
                 "**Gap to %.2f: %.4f.** Class-agnostic test %s. Chain incumbent (secondary): %s. Models trained on "
                 "research_only data: %s."
                 % (h["test_mean"], (" +- %.4f" % h["test_sd"]) if h.get("test_sd") is not None else "", h["seeds"],
                    h["milestone"], h["pool"], h["pool_images"], TARGET, h["gap_to_target"],
                    _f(h.get("agnostic_mean")), _f(h.get("incumbent_test")), h.get("research_only")))
        if h.get("rolled_back"):
            L.append("")
            L.append("That milestone's dev verdict was **%s** and the stream rolled back: its current pool is %s, not %s."
                     % (h.get("verdict"), h.get("current_pool"), h["pool"]))
    else:
        L.append("**Sealed test mAP50-95: not read yet** (test is read only at milestones; milestone 0 is the R0 "
                 "baseline). Target %.2f." % TARGET)
    L.append("")
    if rep["capacity_arms"]:
        L.append("## Capacity arms at R0 (L-4; test read once per arm)")
        L.append("")
        L.append("| Arm | Experiment | Dev mean | Test 12-class | Gap to 0.90 | Chosen |")
        L.append("|---|---|---|---|---|---|")
        for a in rep["capacity_arms"]:
            L.append("| %s | %s | %s | %s | %s | %s |" % (a["arm"], a["exp"], _f(a["dev_mean"]), _ms(a["test"]),
                                                       _f(a["gap_to_target"]), "yes" if a["chosen"] else ""))
        L.append("")
    L.append("## Milestones")
    L.append("")
    L.append("| # | Experiment | Pool (images) | Dev | ImageWeeds (same-lab) | Test | Test agnostic | Incumbent test | "
             "Verdict (dev, 5 v 5) | Gap to 0.90 | research_only |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rep["milestones"]:
        ex = r["exams"]
        inc = (((r.get("incumbent") or {}).get("exams") or {}).get("test") or {}).get("twelve")
        L.append("| %d | %s | %s (%s) | %s | %s | %s | %s | %s | %s%s | %s | %s |"
                 % (r["n"], r["exp"], r["pool"], r.get("pool_images"), _ms((ex.get("dev") or {}).get("twelve")),
                    _ms((ex.get("imageweeds") or {}).get("twelve")), _ms((ex.get("test") or {}).get("twelve")),
                    _ms((ex.get("test") or {}).get("agnostic")), _f(inc), r.get("verdict") or "-",
                    (" (p %.3f)" % r["perm_p"]) if r.get("perm_p") is not None else "", _f(r.get("gap_to_target")),
                    r.get("research_only")))
    L.append("")
    L.append("## Increments")
    L.append("")
    L.append("| Increment | Segment | Images | Sources | Target boxes | Pinned | v3 | P_data | P_recipe | blame | Truth | "
             "Disposition |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in rep["timeline"]:
        L.append("| %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s |"
                 % (t["increment"], t["segment"], t["images"],
                    ", ".join("%s %d" % kv for kv in sorted((t["sources"] or {}).items())),
                    ", ".join("%s %d" % kv for kv in sorted(t["target_boxes"].items())), t["pinned_verdict"] or "-",
                    t["v3_verdict"] or "-", _f(t["p_data"], 3), _f(t["p_recipe"], 3), t["blame"] or "-",
                    t["truth"] or "-", ("%s, then %s" % (t["disposition"], t["status"]) if t["disposition"] and t["status"]
                                        not in (None, t["disposition"]) else t["disposition"] or t["status"])))
    L.append("")
    L.append("## Segments")
    L.append("")
    L.append("| Segment | State | Base pool (images) | M/|P| | K | Recipes | Truth | Base dev | Incumbent dev | Chosen |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in rep["segments"]:
        L.append("| %s | %s | %s (%s) | %s | %d | %s | %s | %s | %s | %s |"
                 % (s["exp"], s["state"], s["base_pool"], s["pool_images"], _f(s["m_over_pool"], 3), s["k"],
                    ",".join(s["recipes"]), "on" if s["truth"] else "off", _ms(s["base_dev"]), _f(s["incumbent_dev"]),
                    s["chosen"] or "-"))
    L.append("")
    L.append("## Yield per source (§7.4)")
    L.append("")
    for src, y in rep["yield"]["stream"].items():
        L.append("- %s: %s%s" % (src, ", ".join("%s %d" % kv for kv in sorted(y.items())),
                                 ("; intake %s" % rep["yield"]["intake"][src]) if src in rep["yield"]["intake"] else ""))
    if not rep["yield"]["stream"]:
        L.append("- nothing cut yet")
    L.append("")
    su = rep["su"]
    L.append("## SU against the envelope (L-2)")
    L.append("")
    window = ("of the %s SU window" % su["window_cap_su"] if su.get("window_cap_su") is not None
              else "SU this month (no monthly window)")
    L.append("Spent %.1f of %s SU (to %s); %s: %.1f %s. %s"
             % (su["spent_su"], su["envelope_su"], su["until"], su["month"], su["month_su"], window, su["note"]))
    L.append("")
    q = rep.get("supply") or {}
    L.append("## Supply")
    L.append("")
    L.append("Eligible target images %s (M = %d); held %s; past deadline %s; OtherPlant-only rows queued %s."
             % (q.get("eligible_images"), rep["M"], q.get("held"), q.get("held_past_deadline"), q.get("other_kind")))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m weed_optimizer_framework.tools.inc2.stream_report",
                                 description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stream")
    g.add_argument("--segment")
    a = ap.parse_args(argv)
    try:
        if a.stream:
            rep = build(a.stream)
            h = rep["headline"]
            print("[inc2.stream_report] %s: %s" % (a.stream, ("test %.4f, gap to 0.90 %.4f" % (h["test_mean"], h["gap_to_target"]))
                                                   if h else "no test read yet"))
        else:
            st = ST.Stream(ST.sid_of_exp(a.segment), quiet=True)
            try:
                st.load()
            except ST.StreamError:
                st = None
            segment_report(a.segment, stream=st)
    except (ST.StreamError, OSError, ValueError, KeyError) as e:
        print("[inc2.stream_report] ERROR: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
