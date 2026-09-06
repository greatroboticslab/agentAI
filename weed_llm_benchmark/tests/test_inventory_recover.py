#!/usr/bin/env python3
"""Unit tests for the inventory evidence-recovery tool (no cluster, no network).

The tool closes a *listing* gap, not an archival one: `results_csv` fills 17
of 162 real cases and `strategy` fills 10, even though a training case that
ran under `run_m1_merged_seeds.sh` almost always left a `results.csv`, a
job-scoped recipe JSON and a job log on shared storage -- the inventory's
`artifact_paths_expected` list simply never named them. These tests build a
small fake storage tree that mirrors that script's real naming conventions
(`RUN_TAG="job$SLURM_JOB_ID"`, the job-scoped `m1_<tier>_seed<N>_<jobid>.json`
that records its own `summary.save_dir`) and check the properties the task
promises:

  * a job id in a file's own name, or in an ancestor run directory's name, is
    proposed at high confidence with the reason stated
  * a recipe JSON matched by job id is opened and its recorded `save_dir` is
    followed to the `results.csv` it names -- the strongest evidence kind
  * a SLURM array task id (`<array>_<task>`) also matches via its array-id
    prefix, and the evidence says so
  * a near-miss job id (a real id embedded in a longer number) never matches,
    because of the same word-boundary rule
  * a file inside a directory this tool must never walk (`downloads/`) is
    never found, even carrying a real job id
  * a file with no id anywhere, whose mtime merely falls inside a job's sacct
    window, is proposed only at low confidence and is excluded from `apply`
    at the default threshold
  * a case with no job id on record proposes nothing and says why
  * the per-case proposal cap is enforced and reported, not silently truncated
  * `apply` only ever writes `artifact_paths_expected`; every other field on
    every case is byte-for-byte the same before and after
  * `apply` is idempotent, and two `scan`s over an unchanged tree agree exactly

Run:  python3 -m pytest tests/test_inventory_recover.py -q
"""
import copy
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.brain import corpus              # noqa: E402
from weed_optimizer_framework.tools.brain import inventory_recover as ir  # noqa: E402


# --------------------------------------------------------------------------
# fixture helpers
# --------------------------------------------------------------------------
def _touch(root, rel, text="", mtime=None):
    p = pathlib.Path(str(root)) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    if mtime is not None:
        os.utime(str(p), (mtime, mtime))
    return p


def _epoch(text):
    return time.mktime(time.strptime(text, "%Y-%m-%dT%H:%M:%S"))


def _entry(**kw):
    base = {
        "case_id": "case", "date": "2026-05-01", "incident": True,
        "correction_class": "config", "actor_who_corrected": "operator",
        "healthy_control": False, "category": "incident",
        "scope": "cluster-pipeline", "version_tag": "v3.25", "notes": "",
        "job_ids": [], "artifact_paths_expected": [], "signals_expected": [],
    }
    base.update(kw)
    return base


_SACCT_TEXT = (
    "$ sacct -j 55555555 --format=JobID,JobIDRaw,JobName,State,Elapsed,Timelimit,"
    "Start,End,Submit,AllocTRES,ExitCode,NodeList -P -X\n"
    "JobID|JobIDRaw|JobName|State|Elapsed|Timelimit|Start|End|Submit|AllocTRES|"
    "ExitCode|NodeList\n"
    "55555555|55555555|m1_merged|TIMEOUT|12:00:18|12:00:00|2026-05-01T00:00:00|"
    "2026-05-01T12:00:00|2026-04-30T23:50:00|gres/gpu=1|0:0|v001\n"
)

WINDOW_START = _epoch("2026-05-01T00:00:00")
WINDOW_END = _epoch("2026-05-01T12:00:00")
INSIDE_WINDOW = _epoch("2026-05-01T06:00:00")
OUTSIDE_WINDOW = _epoch("2020-01-01T00:00:00")


def build_fixture(root):
    """A storage tree + matching spec covering every evidence kind and decoy."""
    # -- case A: job id in a job log's own name (already listed -- must not be
    # re-proposed), a job-scoped recipe JSON that names its own save_dir, and
    # the results.csv that save_dir points to, whose directory name also
    # independently carries the job id.
    _touch(root, "results/framework/m1_merged_m1cur_s2_44767709.out", "log\n")
    _touch(root, "results/framework/m1_curated_seed101_44767709.json",
          json.dumps({"tier": "curated", "seed": 101, "iteration": "m1_curated_s101",
                     "job_id": "44767709",
                     "summary": {"save_dir": "results/framework/"
                                             "mega_iterm1_curated_s101/job44767709"}}))
    _touch(root, "results/framework/mega_iterm1_curated_s101/job44767709/results.csv",
          "epoch,time,metrics/mAP50-95(B)\n1,10.0,0.5\n")
    # decoy: the same recipe's legacy, non-job-scoped run directory -- no job id
    # anywhere in this path, so it must never be proposed for case A.
    _touch(root, "results/framework/mega_iterm1_curated_s101/train/results.csv",
          "epoch,time,metrics/mAP50-95(B)\n1,9.0,0.4\n")
    # decoy: a real job id embedded in a longer number must not match.
    _touch(root, "results/framework/old_144767709.out", "unrelated\n")
    # decoy: a real job id, inside a directory this tool must never walk.
    _touch(root, "downloads/dump/job44767709_extra.out", "must never be found\n")

    case_a = _entry(case_id="case-A-recipe-chain", job_ids=["44767709"],
                    signals_expected=["walltime_bound"],
                    artifact_paths_expected=[
                        "results/framework/m1_merged_m1cur_s2_44767709.out",
                        "sacct -j 44767709",
                    ])

    # -- case B: a collected sacct artifact (high confidence via filename), and
    # one unclaimed file whose mtime alone falls inside that job's window
    # (low confidence, weak on a shared cluster) plus one clearly outside it
    # (must not be proposed at all).
    _touch(root, "results/framework/sacct_55555555.txt", _SACCT_TEXT)
    _touch(root, "results/framework/mega_iterUNRELATED/train/results.csv",
          "epoch,time,metrics/mAP50-95(B)\n1,8.0,0.3\n", mtime=INSIDE_WINDOW)
    _touch(root, "results/framework/mega_iterOUTSIDEWINDOW/train/results.csv",
          "epoch,time,metrics/mAP50-95(B)\n1,7.0,0.2\n", mtime=OUTSIDE_WINDOW)
    case_b = _entry(case_id="case-B-mtime-window", job_ids=["55555555"],
                    correction_class="operational", signals_expected=["pool_growth"])

    # -- case C: a SLURM array task id; the trace file is named after the
    # array's base job id only, per JOB_KEY in run_m1_merged_seeds.sh.
    _touch(root, "results/framework/_brain/weed/trace/m1_curated_s101_44234060.jsonl",
          '{"epoch": 1, "ok": true}\n')
    case_c = _entry(case_id="case-C-array-task", job_ids=["44234060_2"],
                    signals_expected=["gate_noop"])

    # -- case D: enough independent matches to hit the per-case proposal cap.
    for i in range(15):
        _touch(root, "results/framework/cap_test/run77777777_%02d/results.csv" % i,
              "epoch,time,metrics/mAP50-95(B)\n1,%d.0,0.1\n" % i)
    case_d = _entry(case_id="case-D-cap-test", job_ids=["77777777"])

    # -- case E: no job id on record at all -- genuinely nothing to search for.
    _touch(root, "results/framework/framework_results.json", "{}")
    case_e = _entry(case_id="case-E-no-job", job_ids=[],
                    correction_class="design", signals_expected=["plateau"],
                    artifact_paths_expected=["results/framework/framework_results.json"])

    spec = {"domain": "weed", "cases": [case_a, case_b, case_c, case_d, case_e]}
    return spec


def _case(prop, case_id):
    for c in prop["cases"]:
        if c["case_id"] == case_id:
            return c
    raise KeyError(case_id)


def _paths(case_prop):
    return {p["path"] for p in case_prop["proposals"]}


def _evidence_kinds(case_prop, path):
    for p in case_prop["proposals"]:
        if p["path"] == path:
            return {e["kind"] for e in p["evidence"]}
    return set()


def _confidence(case_prop, path):
    for p in case_prop["proposals"]:
        if p["path"] == path:
            return p["confidence"]
    return None


# --------------------------------------------------------------------------
# job id -> filename / dirname / recipe chain
# --------------------------------------------------------------------------
def test_job_id_in_filename_is_proposed_high_confidence(tmp_path):
    spec = build_fixture(tmp_path)
    prop = ir.build_proposal(spec, tmp_path)
    a = _case(prop, "case-A-recipe-chain")
    recipe = "results/framework/m1_curated_seed101_44767709.json"
    assert recipe in _paths(a)
    assert _confidence(a, recipe) >= ir._BAND_HIGH
    assert "job_id_in_filename" in _evidence_kinds(a, recipe)


def test_already_listed_artifact_is_not_reproposed(tmp_path):
    spec = build_fixture(tmp_path)
    prop = ir.build_proposal(spec, tmp_path)
    a = _case(prop, "case-A-recipe-chain")
    assert "results/framework/m1_merged_m1cur_s2_44767709.out" not in _paths(a)


def test_recipe_named_save_dir_reaches_its_results_csv(tmp_path):
    spec = build_fixture(tmp_path)
    prop = ir.build_proposal(spec, tmp_path)
    a = _case(prop, "case-A-recipe-chain")
    csv_path = "results/framework/mega_iterm1_curated_s101/job44767709/results.csv"
    assert csv_path in _paths(a)
    assert _confidence(a, csv_path) >= ir._BAND_HIGH
    # Found twice over -- the job id names its own directory, and the recipe
    # this tool already trusts also names the same file -- and both reasons
    # must survive into one merged proposal, not overwrite each other.
    kinds = _evidence_kinds(a, csv_path)
    assert "job_id_in_dirname" in kinds
    assert "recipe_named_path" in kinds


def test_legacy_run_directory_with_no_job_id_is_not_proposed(tmp_path):
    spec = build_fixture(tmp_path)
    prop = ir.build_proposal(spec, tmp_path)
    a = _case(prop, "case-A-recipe-chain")
    legacy = "results/framework/mega_iterm1_curated_s101/train/results.csv"
    assert legacy not in _paths(a)


def test_near_miss_job_id_never_matches_any_case(tmp_path):
    spec = build_fixture(tmp_path)
    prop = ir.build_proposal(spec, tmp_path)
    for c in prop["cases"]:
        assert "results/framework/old_144767709.out" not in _paths(c)


def test_skip_directory_hides_a_file_even_with_a_real_job_id(tmp_path):
    spec = build_fixture(tmp_path)
    prop = ir.build_proposal(spec, tmp_path)
    for c in prop["cases"]:
        assert not any("downloads/" in p for p in _paths(c))
    # Confirmed at the walk level too: the file was never indexed at all.
    index, _skipped, _stats = ir.walk_storage(tmp_path)
    assert not any("downloads/" in item["rel"] for item in index)


# --------------------------------------------------------------------------
# array task ids
# --------------------------------------------------------------------------
def test_array_task_id_matches_via_its_array_base(tmp_path):
    spec = build_fixture(tmp_path)
    prop = ir.build_proposal(spec, tmp_path)
    c = _case(prop, "case-C-array-task")
    trace = "results/framework/_brain/weed/trace/m1_curated_s101_44234060.jsonl"
    assert trace in _paths(c)
    kinds = _evidence_kinds(c, trace)
    assert "job_id_in_filename" in kinds
    detail = next(p for p in c["proposals"] if p["path"] == trace)["evidence"][0]["detail"]
    assert "44234060" in detail and "44234060_2" in detail


# --------------------------------------------------------------------------
# mtime-window: real, weak, never applied by default
# --------------------------------------------------------------------------
def test_sacct_artifact_matched_high_confidence_by_filename(tmp_path):
    spec = build_fixture(tmp_path)
    prop = ir.build_proposal(spec, tmp_path)
    b = _case(prop, "case-B-mtime-window")
    sacct_path = "results/framework/sacct_55555555.txt"
    assert sacct_path in _paths(b)
    assert _confidence(b, sacct_path) >= ir._BAND_HIGH


def test_mtime_only_match_is_low_confidence(tmp_path):
    spec = build_fixture(tmp_path)
    prop = ir.build_proposal(spec, tmp_path)
    b = _case(prop, "case-B-mtime-window")
    weak = "results/framework/mega_iterUNRELATED/train/results.csv"
    assert weak in _paths(b)
    conf = _confidence(b, weak)
    assert conf is not None and conf < ir._BAND_MEDIUM
    assert "mtime_in_job_window" in _evidence_kinds(b, weak)


def test_file_outside_the_job_window_is_not_proposed(tmp_path):
    spec = build_fixture(tmp_path)
    prop = ir.build_proposal(spec, tmp_path)
    b = _case(prop, "case-B-mtime-window")
    outside = "results/framework/mega_iterOUTSIDEWINDOW/train/results.csv"
    assert outside not in _paths(b)


def test_apply_excludes_low_confidence_by_default_but_includes_on_request(tmp_path):
    spec = build_fixture(tmp_path)
    prop = ir.build_proposal(spec, tmp_path)
    weak = "results/framework/mega_iterUNRELATED/train/results.csv"

    spec1 = copy.deepcopy(spec)
    r1 = ir.apply_proposal(spec1, prop)                 # default threshold
    b1 = next(c for c in spec1["cases"] if c["case_id"] == "case-B-mtime-window")
    assert weak not in b1["artifact_paths_expected"]
    assert r1["report"]["artifacts_skipped_low_confidence"] > 0

    spec2 = copy.deepcopy(spec)
    ir.apply_proposal(spec2, prop, min_confidence=0.1)
    b2 = next(c for c in spec2["cases"] if c["case_id"] == "case-B-mtime-window")
    assert weak in b2["artifact_paths_expected"]


# --------------------------------------------------------------------------
# honesty about absence, and the per-case cap
# --------------------------------------------------------------------------
def test_case_with_no_job_id_proposes_nothing_and_says_why(tmp_path):
    spec = build_fixture(tmp_path)
    prop = ir.build_proposal(spec, tmp_path)
    e = _case(prop, "case-E-no-job")
    assert e["proposals"] == []
    assert any("no job id on record" in n for n in e["notes"])


def test_proposal_cap_is_enforced_and_reported(tmp_path):
    spec = build_fixture(tmp_path)
    prop = ir.build_proposal(spec, tmp_path)
    d = _case(prop, "case-D-cap-test")
    assert len(d["proposals"]) == ir.MAX_PROPOSALS_PER_CASE
    assert d["capped"] is True
    assert any("cap" in n for n in d["notes"])
    assert prop["report"]["cases_capped"] >= 1


def test_bulk_directory_is_skipped_by_size_not_walked(tmp_path, monkeypatch):
    monkeypatch.setattr(ir, "MAX_DIR_ENTRIES", 5)
    for i in range(8):
        _touch(tmp_path, "results/framework/bulkdir/f%02d.out" % i, "x\n")
    index, skipped, stats = ir.walk_storage(tmp_path)
    assert not any("bulkdir" in item["rel"] for item in index)
    assert stats["dirs_skipped_by_size"] == 1
    assert any("bulkdir" in s["path"] and "entries exceeds" in s["reason"]
              for s in skipped)


# --------------------------------------------------------------------------
# truth is read-only
# --------------------------------------------------------------------------
def test_apply_never_touches_any_field_but_artifact_paths_expected(tmp_path):
    spec = build_fixture(tmp_path)
    before = copy.deepcopy(spec["cases"])
    prop = ir.build_proposal(spec, tmp_path)
    result = ir.apply_proposal(spec, prop)
    after = result["spec"]["cases"]

    assert len(before) == len(after)
    for orig, new in zip(before, after):
        assert orig["case_id"] == new["case_id"]
        for key in orig:
            if key == "artifact_paths_expected":
                continue
            assert new[key] == orig[key], "field %r changed on %s" % (key, orig["case_id"])
        # Nothing already recorded is ever removed, and nothing is reordered.
        old_list = orig.get("artifact_paths_expected") or []
        new_list = new.get("artifact_paths_expected") or []
        assert new_list[:len(old_list)] == old_list
    assert result["report"]["artifacts_added"] > 0


def test_apply_adds_no_new_top_level_keys(tmp_path):
    spec = build_fixture(tmp_path)
    prop = ir.build_proposal(spec, tmp_path)
    before_keys = set(spec.keys())
    result = ir.apply_proposal(spec, prop)
    assert set(result["spec"].keys()) == before_keys


def test_field_order_is_preserved(tmp_path):
    spec = build_fixture(tmp_path)
    orig_order = list(next(c for c in spec["cases"]
                           if c["case_id"] == "case-A-recipe-chain").keys())
    spec_path = tmp_path / "inventory.json"
    ir.write_inventory(spec_path, spec)

    prop = ir.build_proposal(corpus.read_json(str(spec_path)), tmp_path)
    result = ir.apply_proposal(corpus.read_json(str(spec_path)), prop)
    ir.write_inventory(spec_path, result["spec"])

    # A full write -> scan -> apply -> write round trip through real JSON
    # files must not reorder a single key; artifact_paths_expected already
    # existed in the fixture entry, and its position must not move just
    # because its contents changed.
    reread = corpus.read_json(str(spec_path))
    a = next(c for c in reread["cases"] if c["case_id"] == "case-A-recipe-chain")
    assert list(a.keys()) == orig_order
    assert len(a["artifact_paths_expected"]) > 1     # something was in fact added


# --------------------------------------------------------------------------
# determinism and idempotency
# --------------------------------------------------------------------------
def test_two_scans_over_an_unchanged_tree_agree_exactly(tmp_path):
    spec = build_fixture(tmp_path)
    p1 = ir.build_proposal(spec, tmp_path)
    p2 = ir.build_proposal(spec, tmp_path)
    assert p1 == p2
    out1 = tmp_path / "p1.json"
    out2 = tmp_path / "p2.json"
    corpus.write_json(out1, p1)
    corpus.write_json(out2, p2)
    assert out1.read_bytes() == out2.read_bytes()


def test_apply_is_idempotent(tmp_path):
    spec = build_fixture(tmp_path)
    prop = ir.build_proposal(spec, tmp_path)
    r1 = ir.apply_proposal(spec, prop)
    snapshot = copy.deepcopy(r1["spec"])
    r2 = ir.apply_proposal(r1["spec"], prop)
    assert r2["spec"] == snapshot
    assert r2["report"]["cases_changed"] == 0
    assert r2["report"]["artifacts_added"] == 0


def test_scan_then_apply_twice_via_files_is_idempotent(tmp_path):
    spec = build_fixture(tmp_path)
    spec_path = tmp_path / "inventory.json"
    ir.write_inventory(spec_path, spec)

    read_spec = corpus.read_json(str(spec_path))
    prop = ir.build_proposal(read_spec, tmp_path)
    prop_path = tmp_path / "proposal.json"
    corpus.write_json(prop_path, prop)

    r1 = ir.apply_proposal(corpus.read_json(str(spec_path)),
                           corpus.read_json(str(prop_path)))
    ir.write_inventory(spec_path, r1["spec"])
    first_bytes = spec_path.read_bytes()

    r2 = ir.apply_proposal(corpus.read_json(str(spec_path)),
                           corpus.read_json(str(prop_path)))
    ir.write_inventory(spec_path, r2["spec"])
    second_bytes = spec_path.read_bytes()

    assert first_bytes == second_bytes
    assert r2["report"]["cases_changed"] == 0


# --------------------------------------------------------------------------
# CLI smoke test (exit codes, no crash on the real subcommand wiring)
# --------------------------------------------------------------------------
def test_cli_scan_apply_report_round_trip(tmp_path):
    spec = build_fixture(tmp_path)
    spec_path = tmp_path / "inventory.json"
    ir.write_inventory(spec_path, spec)
    prop_path = tmp_path / "proposal.json"

    assert ir.main(["scan", "--spec", str(spec_path), "--root", str(tmp_path),
                   "--out", str(prop_path)]) == 0
    assert prop_path.is_file()
    assert ir.main(["report", "--proposal", str(prop_path)]) == 0
    assert ir.main(["apply", "--proposal", str(prop_path), "--spec", str(spec_path)]) == 0

    updated = corpus.read_json(str(spec_path))
    a = next(c for c in updated["cases"] if c["case_id"] == "case-A-recipe-chain")
    assert "results/framework/m1_curated_seed101_44767709.json" in a["artifact_paths_expected"]


if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main([__file__, "-q"]))
