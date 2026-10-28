#!/usr/bin/env python3
"""D31 expects real coverage from the label audit (docs/CONTINUOUS_LOOP.md, Amendment 2026-10-27).

Since 2026-10-27 lever L4 runs inc2.audit_stream, which looks each audited box up in the stream's own batches as
well as Step 1's, and writes "lookup": "step1+step1_stream". D31 reads such a record as follows (and every earlier
reading stands: few judged boxes are no finding, L4 is never called for again):

  units   * a step is judged on its species-labelled boxes (the boxes the audit can give a species verdict), not on
            all its boxes: a step mostly of the non-target class, every species box judged and compared, is
            conclusive (the record's coverage block, else its per_species blocks);
          * a stream-lookup record that judged under D31.audit_expected_coverage (0.9) of a step's species boxes
            names the step in coverage_short, whatever the verdict: under the share it is not judged; over the
            share and under 0.9 the verdict stands and the summary adds the shortfall. The summary splits the
            unjudged species boxes into the audit's known limits (small, no_size, failed, label_unseen,
            in_trusted) and the rest, and calls only the rest a defect of the audit's inputs (never "Step 1's
            crops only");
          * a stream-lookup record of full coverage: no shortfall;
          * the live record of 2026-10-05 (lookup absent: Step 1 only) keeps its not-judged reading, and its summary
            now says the record predates the stream lookup and that a re-audit is a person's decision.
  world   the s001 case (tests/test_stream_ap_stalls.py's l4_world): a stream-lookup audit judging 60 % of each step
          (the rest no_crop_row) raises D31's card naming the coverage as a defect of the audit's inputs, calls for
          no L4; one judging every box (not above its baseline)
          raises none and reads 'labels are not the cause'.
  real    the same with the record the lab run of 2026-10-27 wrote for s001 on full copies of the cluster's files
          (tests/fixtures/audit_stream_s002, sha256 in FIXTURE.json lab_files): every species box judged, labels are
          not the cause on inc0001, inc0003 and inc0004, no card.

Run:  python3 tests/test_stream_ap_audit_coverage.py
"""
import copy
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_stream_ap_stalls as STL  # noqa: E402

from test_stream_ap_world import check  # noqa: E402

W = STL.W
DS = STL.DS
STEPS = ["inc0001", "inc0003", "inc0004"]


def stream_record(judged, compared=None, above=False, species=1364, other=0, coverage_block=True, why="no_crop_row"):
    """The live record of 2026-10-05, as inc2.audit_stream would write it with `judged` species verdicts per step,
    the others not judged for `why`."""
    aud = json.loads(STL.AUDIT_BYTES)
    aud["lookup"] = "step1+step1_stream"
    for s in STEPS:
        a = aud["audits"][s]
        a["boxes"] = species + other
        sp = a["species"]
        sp.update(boxes_judged=judged, above_baseline=above, conflict=0, conflict_rate=0.0)
        sp["baseline_for_mix"]["boxes"] = judged if compared is None else compared
        a["no_embedding_reasons"] = {why: species - judged} if species > judged else {}
        if coverage_block:
            a["coverage"] = {"boxes": species + other, "species_boxes": species, "species_judged": judged,
                             "species_judged_share": round(judged / float(species), 4) if species else None,
                             "otherplant_boxes": other, "otherplant_judged": other,
                             "species_not_judged": {why: species - judged} if species > judged else {}}
        else:
            a.pop("coverage", None)
        if other:
            a["per_species"] = {"Sp": {"verified": judged, "conflict": 0, "unknown": 0, "boxes": species},
                                "Other": {"other_ok": other, "conflict": 0, "boxes": other}}
    return aud


def t_units():
    print("units: D31's coverage readings")
    th = W.LS.load_thresholds()
    check("D31.audit_expected_coverage is pre-registered at 0.9, with its reason",
          th["D31"]["audit_expected_coverage"]["value"] == 0.9
          and "2026-10-27" in th["D31"]["audit_expected_coverage"]["why"])
    exp_cov = th["D31"]["audit_expected_coverage"]["value"]
    share = th["D31"]["audit_min_judged_share"]["value"]

    def read(aud):
        return DS._audit_result(None, STL.FakeEv({"x/audit/label_audit.json": aud}), "x", STEPS, share, exp_cov)[0]

    live = read(json.loads(STL.AUDIT_BYTES))
    said = DS._audit_said("x", live)
    check("the live record (Step 1 only): not judged, predates the stream lookup, no coverage shortfall",
          live["verdict"] == "not_judged" and live["predates_stream"] is True and live["coverage_short"] == []
          and live["stream_lookup"] is False and all(s["species_boxes"] == 1364 for s in live["steps"].values()),
          live)
    check("  its summary keeps the crops explanation and says a re-audit is a person's decision",
          "Step 1's crops" in said and "predates it" in said and "person's decision" in said
          and "remote.py submit audit refuses" in said, said)
    full = read(stream_record(1364))
    check("a stream-lookup record judging every species box, not above: labels not the cause, no shortfall",
          full["verdict"] == "not_labels" and full["coverage_short"] == [] and full["stream_lookup"] is True
          and not full["predates_stream"], full)
    check("  its summary names no shortfall", "short of the expected" not in DS._audit_said("x", full))
    few = read(stream_record(300))
    said = DS._audit_said("x", few)
    check("a stream-lookup record judging 300 of 1,364: not judged, every step short of 0.9",
          few["verdict"] == "not_judged" and few["coverage_short"] == STEPS, few)
    check("  its summary: a defect of the audit's inputs, not 'Step 1's crops only', not 'predates'",
          "defect of its inputs" in said and "Step 1's crops only" not in said and "predates" not in said
          and "inc0001 300 of 1364" in said and "no_crop_row 3192" in said and "known limits" not in said, said)
    lim = read(stream_record(300, why="small"))
    said = DS._audit_said("x", lim)
    check("  short for boxes the intake left out (small): named, as the audit's known limits, not a defect",
          lim["coverage_short"] == STEPS and "known limits of the audit" in said and "small 3192" in said
          and "so this is a defect" not in said, said)
    uns = read(stream_record(1000, why="label_unseen"))
    said = DS._audit_said("x", uns)
    check("  short for species the probe never saw (label_unseen): the verdict stands, known limits, not a defect",
          uns["verdict"] == "not_labels" and uns["coverage_short"] == STEPS and "label_unseen 1092" in said
          and "known limits of the audit" in said and "so this is a defect" not in said, said)
    mix = stream_record(1000, why="small")
    for s_ in STEPS:
        mix["audits"][s_]["coverage"]["species_not_judged"] = {"small": 200, "image_not_in_crops": 164}
    said = DS._audit_said("x", read(mix))
    check("  short for both: the known limits and the defect each named for what they are",
          "image_not_in_crops 492: the audit read Step 1's and the stream's crops, so this is a defect" in said
          and "small 600: known limits" in said, said)
    old_t = stream_record(1000, coverage_block=False, why="no_crop_row")
    for s_ in STEPS:
        old_t["audits"][s_]["no_embedding_reasons"] = {}
        old_t["audits"][s_]["boxes_by_status"] = dict(old_t["audits"][s_].get("boxes_by_status") or {},
                                                      in_trusted=364)
    said = DS._audit_said("x", read(old_t))
    check("  a record without a coverage block: the in_trusted boxes of its statuses read as a known limit",
          "in_trusted 1092: known limits" in said and "so this is a defect" not in said, said)
    mid = read(stream_record(1000))
    said = DS._audit_said("x", mid)
    check("a stream-lookup record judging 1,000 of 1,364 (over the share, under 0.9): the verdict stands "
          "(labels not the cause) and the shortfall is named", mid["verdict"] == "not_labels"
          and mid["coverage_short"] == STEPS and "not the cause" in said and "short of the expected 0.9" in said,
          (mid["verdict"], said))
    above = read(stream_record(1000, above=True))
    check("  and above its baseline on them: labels (X3), the shortfall named",
          above["verdict"] == "labels" and above["coverage_short"] == STEPS
          and "short of the expected" in DS._audit_said("x", above), above["verdict"])
    oth = read(stream_record(300, species=300, other=1064))
    check("a step of 300 species and 1,064 non-target boxes, every species box judged and compared: conclusive "
          "(the share is of the species-labelled boxes), no shortfall", oth["verdict"] == "not_labels"
          and all(s["conclusive"] and s["species_boxes"] == 300 for s in oth["steps"].values())
          and oth["coverage_short"] == [], oth)
    oth2 = read(stream_record(300, species=300, other=1064, coverage_block=False))
    check("  the same read from the per_species blocks (a record without a coverage block)",
          oth2["verdict"] == "not_labels" and all(s["species_boxes"] == 300 for s in oth2["steps"].values()), oth2)
    step1 = stream_record(300)
    step1.pop("lookup")
    r = read(step1)
    check("a Step-1-only record judging 300 of 1,364 (no image_not_in_crops): not judged, no coverage shortfall "
          "expected of it, not 'predates'", r["verdict"] == "not_judged" and r["coverage_short"] == []
          and not r["predates_stream"], r)


def t_world():
    print("world: D31 on s001 with a stream-lookup audit")
    w, exp, pid = STL.l4_world("cov_short")
    STL.write_audit(w, exp, json.dumps(stream_record(818)).encode())      # 60 % of each step
    n0 = len(w.ledger())
    w.tick(4)
    led = STL.led_since(w, n0)
    d31 = STL.diag(w, "D31")
    res = ((d31.get("detail") or {}).get("audits") or {}).get(exp) or {}
    check("D31 reads it: labels not the cause, every step short of 0.9, OP_CARD, no L4",
          res.get("verdict") == "not_labels" and res.get("coverage_short") == STEPS
          and "OP_CARD" in (d31.get("levers") or []) and "L4" not in (d31.get("levers") or [])
          and (d31.get("detail") or {}).get("coverage_short") == [exp], (d31.get("levers"), res.get("verdict")))
    cards = [e for e in led if e.get("event") == "card" and "D31" in str(e.get("title"))]
    check("one card names the coverage shortfall", len(cards) == 1 and "short of the expected" in cards[0].get(
        "detail", "") and "defect of its inputs" in cards[0]["detail"], [c.get("detail", "")[:200] for c in cards])
    check("no L4 proposed again", not [e for e in led if e.get("lever") == "L4" and e.get("event") == "proposed"])

    w2, exp2, _pid2 = STL.l4_world("cov_full")
    STL.write_audit(w2, exp2, json.dumps(stream_record(1364)).encode())
    n0 = len(w2.ledger())
    w2.tick(4)
    led = STL.led_since(w2, n0)
    d31 = STL.diag(w2, "D31")
    res = ((d31.get("detail") or {}).get("audits") or {}).get(exp2) or {}
    check("a stream-lookup audit of every box, not above: labels not the cause, no card, nothing called for",
          res.get("verdict") == "not_labels" and not res.get("coverage_short")
          and not (set(d31.get("levers") or []) & {"L4", "X3", "OP_CARD"})
          and not [e for e in led if e.get("event") == "card" and "D31" in str(e.get("title"))],
          (d31.get("levers"), res.get("verdict")))


def t_real():
    print("real: D31 on s001 with the lab run's audit of inc0001, inc0003, inc0004 (the cluster's files, copied)")
    fx = pathlib.Path(__file__).resolve().parent / "fixtures" / "audit_stream_s002"
    rec = json.loads((fx / "FIXTURE.json").read_text())["lab_files"]["label_audit_s001_lab_2026-10-27.json"]
    raw = (fx / "label_audit_s001_lab_2026-10-27.json").read_bytes()
    check("the lab record is the fixture's (sha256 %s...)" % rec["sha256"][:12],
          __import__("hashlib").sha256(raw).hexdigest() == rec["sha256"])
    aud = json.loads(raw)
    check("it judged every species box of each step (1,364 of 1,364) with the stream lookup",
          aud["lookup"] == "step1+step1_stream" and all(aud["audits"][s]["coverage"]["species_judged"] == 1364
                                                        for s in STEPS))
    w, exp, _pid = STL.l4_world("cov_real")
    STL.write_audit(w, exp, raw)
    n0 = len(w.ledger())
    w.tick(4)
    led = STL.led_since(w, n0)
    d31 = STL.diag(w, "D31")
    res = ((d31.get("detail") or {}).get("audits") or {}).get(exp) or {}
    check("D31: judged, labels are not the cause on inc0001, inc0003, inc0004; no shortfall, no card, no L4, no X3",
          res.get("verdict") == "not_labels" and not res.get("coverage_short")
          and all(s["conclusive"] and s["above_baseline"] is False for s in (res.get("steps") or {}).values())
          and not (set(d31.get("levers") or []) & {"L4", "X3", "OP_CARD"})
          and not [e for e in led if e.get("event") == "card" and "D31" in str(e.get("title"))],
          (d31.get("levers"), res.get("verdict"), d31.get("summary")))
    check("  its summary says so", "labels are not the cause" in str(d31.get("summary")), d31.get("summary"))


def main():
    try:
        for fn in (t_units, t_world, t_real):
            try:
                fn()
            except Exception as e:               # noqa: BLE001
                import traceback
                check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e,
                                                                     traceback.format_exc()[-2500:]))
    finally:
        shutil.rmtree(str(W.TMP), ignore_errors=True)
    print("\n%d failure(s)" % len(W.FAILURES))
    for x in W.FAILURES:
        print("  FAILED: %s" % x)
    return 1 if W.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
