#!/usr/bin/env python3
"""V3F deviation 5 (2026-10-24; docs/prereg/V3F_dev5_2026-10-24.md): the family's second continuation verdict end to
end on a synthetic family (inc2.v3fam_verdict.verdict, the same computation as the verdict of 2026-10-13 and its
continuation of 2026-10-22), as the cluster's records stood on 2026-10-24:

- the verdict of 2026-10-13 written while seven arms had never started, its continuation of 2026-10-22 written with
  D8, D4F, D4N, D4R15, D1 and D5B 'failed twice' from the ticker's --ticker-failed (the refusals of the platform's
  stale records), D6 'not run (walltime)' from its smoke record; both pinned by sha256 as the code pins the cluster's;
- the reads follow the second continuation and refuse before it; the verb refuses while any of the six arms has no
  record since the reopening, writing no verdict (its first step moves the superseded failure records aside);
- D6's smoke of 2026-10-21 (its epoch time the whole fit) superseded: D6 waits too, and the verb's first step moves
  that record aside; the ticker's failure of D5B with only its jobs of 2026-10-21 ends nothing;
- once the arms are rescored (D5B failing again after the reopening, a job of it since; D6 scored): written once
  under its own names, every earlier record unchanged byte for byte, every comparison either earlier verdict decided
  reproduced (P1 and P4 equal), P2, P3 and P5 decided; a second run refuses;
- an attempt cut off before its verdict file left partial records: moved aside, then every record written once;
- a continuation of 2026-10-22 the second continuation does not reproduce (P1's D 1e-6 off): refused, nothing written.
Slow (four verdicts with the bootstraps): the mutation run uses it only for the verb's own mutants.

Run:  python3 tests/test_v3fam_dev5_verdict.py
"""
import functools
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_v3fam_verdict as VT  # noqa: E402  (sets INC_DIR / REPO in a temporary directory first)

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc2 import twostage as TS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam as F  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_systems as SYS  # noqa: E402
from weed_optimizer_framework.tools.inc2 import v3fam_verdict as VR  # noqa: E402

FAILURES = VT.FAILURES
check = VT.check
refused = VT.refused
SIX = ("D8", "D4F", "D4N", "D4R15", "D1", "D5B")
LATE = ("D4F", "D4N", "D4R15", "D6", "D8S", "D8M")
FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "v3fam_dev5"


def names_of(d):
    return [d["verdict"], d["external"], d["ext"], d["report"] + ".json", d["report"] + ".md"]


def t_second_continuation():
    print("the second continuation verdict (deviation 5) on a synthetic family")
    try:
        from weed_optimizer_framework.tools.inc2 import v3fam_supersede as SUP
    except ImportError:
        SUP = None
    VT.reset()
    VT.build_manifests()
    VT.build_scores(skip=LATE + ("D5B", "D1"))
    for a in ("D4F", "D4N", "D4R15", "D6", "D8", "D1"):
        if SYS.score_record_path(a).exists():
            SYS.score_record_path(a).unlink()
    smoke = F.root() / "d6" / "smoke" / "smoke.json"
    smoke.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(str(FIX / "d6" / "smoke.json"), str(smoke))      # job 49542653's record: not run (walltime)
    real = (F.comparison, VR.verdict_ready, VR.PARTIAL_VERDICT_SHA256, getattr(VR, "CONT_SHA256", None),
            dict(getattr(VR, "DEV5_SUPERSEDED_FAILED", {}) or {}), getattr(SUP, "ENTRIES", None))
    F.comparison = functools.partial(real[0], resamples=VT.RESAMPLES)
    made = []
    cap = C.INC_DIR / "capacity"
    smoke_sha = F._sha(smoke)
    if SUP is not None:
        SUP.ENTRIES = ()                                    # the records as written then: nothing superseded yet
    try:
        VR.verdict_ready = lambda *a, **k: (True, "")
        v1 = VR.verdict()                                  # the verdict of 2026-10-13, as written then
        VR.verdict_ready = real[1]
        VR.PARTIAL_VERDICT_SHA256 = F._sha(VR.verdict_path())
        made = VT.family_ready()
        six = ["score:%s" % a for a in SIX]
        cont = VR.verdict(ticker_failed=six)               # the continuation of 2026-10-22: six arms 'failed twice'
        VR.CONT_SHA256 = F._sha(cap / VR.CONTINUATION["verdict"])
        sup = {"v3fam/v1/failed/score_%s.json" % a: F._sha(VR.failed_file("score:%s" % a)) for a in SIX}
        if SUP is not None:
            # deviation 5's entries on the synthetic family: its six failure records, and D6's smoke of 2026-10-21
            # (job 49542653's record byte for byte, its epoch time the whole fit)
            SUP.ENTRIES = tuple((r, s, SUP.DEVIATION_5, SUP.FOLDER_5, "the synthetic family's") for r, s in sup.items()) \
                + ((SUP.D6_SMOKE, smoke_sha, SUP.DEVIATION_5, SUP.FOLDER_5, "the smoke of 2026-10-21"),)
            VR.DEV5_SUPERSEDED_FAILED = {"score:%s" % a: sup["v3fam/v1/failed/score_%s.json" % a] for a in SIX}
        before = {n: (cap / n).read_bytes() for n in [VR.VERDICT, VR.EXTERNAL, VR.EXT, VR.REPORT + ".json",
                                                      VR.REPORT + ".md"] + names_of(VR.CONTINUATION)}
        check("the continuation of 2026-10-22 as written then: P2, P3 and P5 'not run', the six arms 'failed (... failed "
              "twice)', D6 'not run (walltime)'", cont["primaries"]["P2"]["status"] == "not run"
              and all(cont["arms"][a].startswith("failed") for a in SIX)
              and cont["arms"]["D6"] == "not run (walltime)", cont["arms"])
        e_read = refused(VR.read_verdict)
        st = {a: VR.arm_status(a) for a in SIX + ("D6",)}
        check("the reads follow the second continuation and refuse until it is decided, naming deviation 5; the six "
              "arms read 'missing' (their failure records of 2026-10-22 not read), and D6 too (its smoke of 2026-10-21 "
              "superseded: taken again)", VR.family_verdict_path().name == "v3fam_v1_cont2.json" and e_read is not None
              and "deviation 5" in str(e_read) and all(v == "missing" for v in st.values()), (e_read, st))
        e_wait = refused(VR.verdict)
        moved = sorted(p.name for p in (VR.failed_dir() / "superseded_2026-10-24").glob("score_*.json"))
        smoke_moved = (smoke.parent / "superseded_2026-10-24" / "smoke.json")
        check("  the verb refuses while the six arms and D6 have no record since the reopening, naming them and "
              "deviation 5; no record of the second continuation written, the earlier ones unchanged; its first step "
              "moved the six failure records and D6's smoke of 2026-10-21 aside byte for byte", e_wait is not None
              and "deviation 5" in str(e_wait) and all(a in str(e_wait) for a in SIX + ("D6",))
              and not any((cap / n).exists() for n in names_of(VR.CONTINUATION_2))
              and all((cap / n).read_bytes() == b for n, b in before.items())
              and moved == sorted("score_%s.json" % a for a in SIX) and not smoke.exists()
              and smoke_moved.is_file() and F._sha(smoke_moved) == smoke_sha, (e_wait, moved))
        for name in ("D4F", "D4N", "D4R15", "D8S", "D8M", "D6"):
            for sd in F.SEEDS:
                VT.put(name, "dev", sd, VT.SHIFT[name], arm=VR.SIDES[name][3])
        for a in ("D4F", "D4N", "D4R15", "D8", "D1", "D6"):
            TS._write_json(SYS.score_record_path(a), {"format": SYS.FMT_SCORE_RECORD, "arm": a, "status": "complete",
                                                     "exam": "dev", "dev": []})
        e_d5b = refused(VR.verdict)
        check("  five arms rescored, D5B not: still refused, naming D5B only", e_d5b is not None
              and str(e_d5b).endswith("D5B"), e_d5b)
        (cap / VR.CONTINUATION_2["external"]).write_text('{"partial": "an attempt cut off before its verdict file"}')
        e_old = refused(VR.verdict, ticker_failed=["score:D5B"])
        check("  D5B named failed by the ticker while its only score jobs ran before the reopening (the list of "
              "2026-10-22 again): refused, naming it, nothing written", e_old is not None
              and "no job of that item since the reopening" in str(e_old) and "score:D5B" in str(e_old)
              and not (cap / VR.CONTINUATION_2["verdict"]).exists(), e_old)
        pv = C.INC_DIR / "_campaign" / "provenance" / "v3fam_score_d5b.json"
        prv = json.loads(pv.read_text()) if pv.exists() else {"attempts": []}
        prv["attempts"].append({"job_id": "5", "status": "build_failed", "started_utc": "2026-10-24T06:00:00Z",
                                "finished_utc": "2026-10-24T06:10:00Z"})
        pv.parent.mkdir(parents=True, exist_ok=True)
        pv.write_text(json.dumps(prv))
        made.append(pv)
        c2 = VR.verdict(ticker_failed=["score:D5B"])          # D5B failed twice again after the reopening
        c = c2["continues"]
        part = sorted(p.name for p in cap.glob("v3fam_v1_cont2_partial_*"))
        check("the second continuation: every record written once under its own names (the partial record of a cut-off "
              "attempt moved aside first); the verdict of 2026-10-13 and the continuation of 2026-10-22 (with their "
              "external, ext and report files) unchanged byte for byte",
              all((cap / n).exists() for n in names_of(VR.CONTINUATION_2)) and len(part) == 1
              and (cap / part[0] / VR.CONTINUATION_2["external"]).exists()
              and all((cap / n).read_bytes() == b for n, b in before.items()), part)
        want = sorted(set(c["reproduced_by_record"]["2026-10-13"]) | set(c["reproduced_by_record"]["2026-10-22"]))
        check("  it names both earlier records by sha256 and deviation 5, reproduces every comparison either decided "
              "(P1 and P4 equal), keeps every family-confirmed primary; P2 and P3 decided now",
              c["verdict"] == VR.VERDICT and c["continuation"] == VR.CONTINUATION["verdict"]
              and c["continuation_sha256"] == VR.CONT_SHA256 and c["deviation"] == VR.DEVIATION_5
              and c["reproduced"] == want and "P1" in want and "P4" in want
              and c2["primaries"]["P1"]["decision"] == cont["primaries"]["P1"]["decision"]
              == v1["primaries"]["P1"]["decision"] and c2["primaries"]["P4"]["decision"] ==
              v1["primaries"]["P4"]["decision"] and c2["primaries"]["P2"]["status"] == "decided"
              and c2["primaries"]["P3"]["status"] == "decided", (c.get("reproduced"), want))
        check("  P5 decided now (D6 scored after its smoke was taken again); D5B 'failed (score:D5B failed twice)' from "
              "the failure the ticker gave after the reopening (a job of it since); the rescored arms complete",
              c2["primaries"]["P5"]["status"] == "decided" and c2["arms"]["D6"] == "complete"
              and c2["arms"]["D5B"] == "failed (score:D5B failed twice)"
              and all(c2["arms"][a] == "complete" for a in ("D8", "D4F", "D4N", "D4R15", "D1")), c2["arms"])
        e_twice = refused(VR.verdict)
        check("  written once: a second run refuses", e_twice is not None and "second continuation" in str(e_twice),
              e_twice)
        # a continuation of 2026-10-22 the second continuation does not reproduce (in a copy of the world: nothing of
        # the second continuation is kept from the run above)
        for n in names_of(VR.CONTINUATION_2):
            (cap / n).unlink()
        doc = json.loads(before[VR.CONTINUATION["verdict"]])
        doc["primaries"]["P1"]["decision"]["diff"] += 1e-6
        (cap / VR.CONTINUATION["verdict"]).write_text(json.dumps(doc))
        VR.CONT_SHA256 = F._sha(cap / VR.CONTINUATION["verdict"])
        e_doc = refused(VR.verdict, ticker_failed=["score:D5B"])
        check("  a continuation of 2026-10-22 whose P1 D the second continuation does not give (1e-6 off): refused, "
              "naming it, nothing written", e_doc is not None and "does not reproduce" in str(e_doc)
              and "2026-10-22" in str(e_doc) and "P1" in str(e_doc)
              and not any((cap / n).exists() for n in names_of(VR.CONTINUATION_2)), e_doc)
        (cap / VR.CONTINUATION["verdict"]).write_bytes(before[VR.CONTINUATION["verdict"]])
        VR.CONT_SHA256 = "0" * 64
        e_other = refused(VR.verdict)
        check("  another continuation in place (not the one of 2026-10-22): written once, refused", e_other is not None
              and "continuation verdict is written once (deviation 4)" in str(e_other), e_other)
    finally:
        F.comparison, VR.verdict_ready, VR.PARTIAL_VERDICT_SHA256 = real[:3]
        if real[3] is not None:
            VR.CONT_SHA256 = real[3]
        if hasattr(VR, "DEV5_SUPERSEDED_FAILED"):
            VR.DEV5_SUPERSEDED_FAILED = real[4]
        if SUP is not None:
            SUP.ENTRIES = real[5]
        VT.unmake(made)
        VT.reset()


def main():
    try:
        try:
            t_second_continuation()
        except Exception as e:                   # noqa: BLE001
            import traceback
            check("t_second_continuation ran", False, "%s: %s\n%s" % (type(e).__name__, e,
                                                                      traceback.format_exc()[-1500:]))
    finally:
        shutil.rmtree(str(VT.TMP), ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
