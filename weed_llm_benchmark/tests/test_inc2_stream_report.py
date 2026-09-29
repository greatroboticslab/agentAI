#!/usr/bin/env python3
"""The stream's reports (inc2/stream_report.py; docs/CONTINUOUS_LOOP.md §3.7).

The world and the synthetic executor are tests/test_inc2_stream.py's (imported
as a library; nothing of it runs). FakeBackend runs every spec; scores are
known, so every number in a report is checked against the score files.

Pinned:
  * before any test read the stream report says so, and leads with it;
  * with milestone 0 only (the R0 baseline), the report leads with its sealed
    test mean +- sd over 5 seeds and the gap to 0.90, the capacity arms
    table carries each arm's test and gap (L-4's read);
  * after a segment and milestone 1, the report leads with milestone 1's test
    (12-class, the species_map50_95 of its final runs), the gap, the
    class-agnostic test and the chain incumbent's secondary test; the
    milestone table (dev, ImageWeeds same-lab, test, verdict and p, gap,
    research_only); the increment timeline (sources, target boxes, pinned
    and v3 verdicts, P_data, P_recipe, blame, truth, disposition); pool size
    and M / |P| per segment; yield per source; SU spent against the L-2
    envelope from the runs' seconds; supply;
  * the segment report: the pinned report's conventions on the segment's
    exams only (dev, imageweeds), the stream commit, GPU-hours;
  * the milestone's RESEARCH_LOG entry: what changed, why, how verified, the
    result with the gap;
  * the CLI (--stream, --segment).

Run:  python3 tests/test_inc2_stream_report.py
"""
import json
import os
import pathlib
import shutil
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_stream as W  # noqa: E402  (sets INC_DIR to its own temporary directory)

from weed_optimizer_framework.tools.inc import driver as D  # noqa: E402
from weed_optimizer_framework.tools.inc2 import stream as ST  # noqa: E402
from weed_optimizer_framework.tools.inc2 import stream_report as SR  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def final_values(exp, exam, key="species_map50_95", rid_fmt="final__base__s%d", seeds=(0, 1, 2, 3, 4)):
    vals = []
    for s in seeds:
        sc = json.loads(D.Paths(exp).score(rid_fmt % s, exam).read_text())
        vals.append(sc[key])
    return vals


def main():
    base, base_rows = W.make_base()
    dev_paths = [W.WORLD / "dev" / ("dev_%d.png" % i) for i in range(2)]
    for i, p in enumerate(dev_paths):
        W.save_pattern(W.pattern(4242 + i), p)
    guard = W.make_guard(dev_paths, base_rows)
    M = 6
    ex = W.Executor(M)
    fb = D.FakeBackend()
    bdl = W.BaselineDouble(fb)

    print("before any test read")
    s0 = W.Step1World("s1_rep0")
    s0.commit()
    st0 = W.new_stream("rep0", base, M, s0, guard, fb=fb)
    rep0 = SR.build("rep0", stream=st0)
    md0 = st0.p.report_md.read_text()
    check("no milestone: the headline says the test has not been read (and the target)",
          rep0["headline"] is None and "not read yet" in md0.splitlines()[2] and "0.90" in md0.splitlines()[2],
          md0.splitlines()[:3])

    print("milestone 0 and the capacity arms")
    for exp, arm in (("rep_m0", "n640"), ("rep_s640", "s640")):
        bdl(["build", "--exp", exp, "--manifest", str(base), "--seeds", "0,1,2,3,4", "--arm", arm, "--role", "b_v2",
             "--final-exams", "dev,imageweeds,test", "--testing"])
        W.drive(exp, fb, ex)
    s1 = W.Step1World("s1_rep")
    mixed = [1, 1, 0, 0, 0, 1] + [0] * 6
    s1.add("good_r", "b0001", mixed, n=M, capture_group="g")
    s1.add("bad_r", "b0002", mixed, n=M, capture_group="b")
    s1.commit()
    sec, sub = W.SecondaryDouble(), W.SubmitDouble(ex)
    st = W.new_stream("rep", base, M, s1, guard, fb=fb, milestone0="rep_m0", baseline=bdl, secondary=sec,
                      submitter=sub)
    cap = W.TMP / "rep_capacity.json"
    cap.write_text(json.dumps({"format": "inc2-capacity/1", "chosen_arm": "n640", "chosen_exp": "rep_m0",
                               "qualifying": [], "n_images": len(base_rows), "m": M, "truth_every": 1,
                               "step_cost": {"gpu_h": [1.0, 1.0], "estimate": True},
                               "arms": {"rep_m0": {"arm": "n640", "mean": 0.40, "sd": 0.001},
                                        "rep_s640": {"arm": "s640", "mean": 0.40, "sd": 0.001, "qualifies": False}}}))
    st.choose_arm(cap)
    rep = SR.build("rep", stream=st)
    t0 = final_values("rep_m0", "test")
    h = rep["headline"]
    check("milestone 0 heads the report: its test mean +- sd over 5 seeds and the gap to 0.90",
          h and h["milestone"] == "rep_m0" and abs(h["test_mean"] - statistics.fmean(t0)) < 1e-12
          and abs(h["test_sd"] - statistics.stdev(t0)) < 1e-12 and abs(h["gap_to_target"] - (0.90 - h["test_mean"])) < 1e-12
          and h["seeds"] == 5, h)
    arms = {a["arm"]: a for a in rep["capacity_arms"]}
    check("the capacity arms table: each arm's test and gap (the R0 read), the chosen arm marked",
          set(arms) == {"n640", "s640"} and arms["n640"]["chosen"] and not arms["s640"]["chosen"]
          and abs(arms["s640"]["test"]["mean"] - statistics.fmean(final_values("rep_s640", "test"))) < 1e-12
          and arms["s640"]["gap_to_target"] is not None, rep["capacity_arms"])

    print("a segment and milestone 1")
    exp1 = st.build(2)
    W.drive(exp1, fb, ex)
    st.commit(exp1)
    seg = json.loads((D.Paths(exp1).root / "report.json").read_text())
    check("the segment report: the segment's exams only, the pinned report's step rows and final groups, the "
          "stream commit, GPU-hours",
          seg["exams"] == ["dev", "imageweeds"] and len(seg["steps"]) == 2
          and [r["model"] for r in seg["final"]][0] == "chain r0: final incumbent"
          and set(seg["final"][0]["exams"]) == {"dev", "imageweeds"} and seg["stream_commit"]["chosen"] == "r0"
          and seg["gpu_hours_total"] > 0 and "test" not in json.dumps(seg["final"]), seg["exams"])
    md_seg = (D.Paths(exp1).root / "report.md").read_text()
    check("the segment report's markdown: steps, the commit's v3 reading, the final table",
          "## Steps" in md_seg and "## Commit (Protocol v3 reading)" in md_seg and "## Final table" in md_seg)
    st.milestone()
    m1 = ST.milestone_exp("rep", 1)
    W.drive(m1, fb, ex)
    st.milestone()
    rep = SR.build("rep", stream=st)
    h = rep["headline"]
    t1 = final_values(m1, "test")
    a1 = final_values(m1, "test", key="agnostic_map50_95")
    inc_test = json.loads(D.Paths(m1).score(ST.SECONDARY_RUN, "test").read_text())["species_map50_95"]
    check("milestone 1 heads the report now: test mean +- sd, the gap, the agnostic test, the incumbent's test",
          h["milestone"] == m1 and abs(h["test_mean"] - statistics.fmean(t1)) < 1e-12
          and abs(h["gap_to_target"] - (0.90 - statistics.fmean(t1))) < 1e-12
          and abs(h["agnostic_mean"] - statistics.fmean(a1)) < 1e-12 and abs(h["incumbent_test"] - inc_test) < 1e-12,
          h)
    md = st.p.report_md.read_text().splitlines()
    first = next(ln for ln in md[1:] if ln.strip())
    check("the markdown leads with the test metric and the gap to 0.90",
          first.startswith("**Sealed test mAP50-95") and "Gap to 0.90: %.4f" % h["gap_to_target"] in first, first)
    rows = {r["exp"]: r for r in rep["milestones"]}
    r1 = rows[m1]
    check("the milestone table: dev, ImageWeeds, test mean +- sd, the verdict and p, the gap, research_only",
          set(rows) == {"rep_m0", m1} and r1["verdict"] == "helps" and r1["perm_p"] is not None
          and abs(r1["exams"]["imageweeds"]["twelve"]["mean"] - statistics.fmean(final_values(m1, "imageweeds"))) < 1e-12
          and abs(r1["exams"]["dev"]["twelve"]["mean"] - statistics.fmean(final_values(m1, "dev"))) < 1e-12
          and r1["gap_to_target"] == h["gap_to_target"] and "research_only" in r1, r1)
    tl = rep["timeline"]
    check("the timeline: every increment with sources, target boxes, pinned and v3 verdicts, P_data, P_recipe, "
          "blame, truth and disposition",
          [t["disposition"] for t in tl] == ["accepted", "data"] and all(t["pinned_verdict"] and t["v3_verdict"]
                                                                       for t in tl)
          and tl[1]["blame"] in ("data", "recipe") and tl[0]["truth"] == "helps"
          and tl[0]["target_boxes"] == {"Waterhemp": M, "MorningGlory": M, "Ragweed": M}
          and tl[0]["sources"] == {"good_r": M}, tl)
    sg = rep["segments"][0]
    check("pool size, M / |P| and dev per segment",
          sg["base_pool"] == "P_0" and sg["pool_images"] == len(base_rows)
          and abs(sg["m_over_pool"] - M / float(len(base_rows))) < 1e-12 and sg["base_dev"]["n"] == 3
          and sg["incumbent_dev"] is not None, sg)
    y = rep["yield"]["stream"]
    check("yield per source: images cut and their dispositions",
          y == {"good_r": {"cut": M, "accepted": M}, "bad_r": {"cut": M, "data": M}}, y)
    hours = 0.0
    for e in (exp1, m1):
        stj = json.loads(D.Paths(e).state.read_text())
        hours += 100.0 * len(stj["runs"]) / 3600.0
    su = rep["su"]
    check("SU against the L-2 envelope: the runs' seconds of every stream experiment",
          abs(su["spent_su"] - hours) < 1e-9 and su["envelope_su"] == 1000 and su["window_cap_su"] == 350, su)
    check("supply: the eligible queue and holds", rep["supply"]["eligible_images"] == 0
          and "held" in rep["supply"], rep["supply"])
    entry = (st.p.milestone_dir(1) / "research_log_entry.md").read_text()
    check("the RESEARCH_LOG entry of milestone 1: the pool change, the rule, the 5 v 5 dev decision, the test "
          "and gap", "P_0 (%d images) to P_1 (%d images)" % (len(base_rows), len(base_rows) + M) in entry
          and "Verdict: helps" in entry and ("%.4f" % statistics.fmean(t1)) in entry
          and ("gap to 0.90: %.4f" % (0.90 - statistics.fmean(t1))) in entry, entry)
    rc = SR.main(["--stream", "rep"])
    rc2 = SR.main(["--segment", exp1])
    check("the CLI: --stream and --segment exit 0", rc == 0 and rc2 == 0)
    check("an unknown stream is an error, not a traceback", SR.main(["--stream", "nosuch"]) == 1)

    print("\n%d failure(s)" % len(FAILURES))
    if not FAILURES:
        shutil.rmtree(W.TMP, ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    os.environ.setdefault("INC_SCORER_TESTING", "1")
    sys.exit(main())
