#!/usr/bin/env python3
"""The INC autopilot's evidence is dev-only by construction, and every value
in it has an address that resolves (docs/INC_AUTOPILOT.md, (a) 1 and (d)).

Pinned:
  * the allow-list: the experiment and Step 1 artifacts named in
    evidence.py are loadable; a score file, a run directory, a manifest, the
    splits lock, a reserved directory or a path escape is refused, and a
    refused name's content never reaches the evidence;
  * load_dir opens only allow-listed files: with decoy score files (test,
    ood22, imageweeds), a test manifest and select_clusters.csv planted in
    the snapshot tree, the one file-opening function never sees them;
  * the scrub: report.json keeps only the dev column of its final table
    (5 rows x 4 non-dev exams dropped, each pointer recorded); a stamp dict
    of a non-dev exam and a non-dev score path are dropped, a list keeps its
    positions (None in place), and every pointer into the scrubbed copy
    addresses the same value in the raw object; leaks() is empty after it;
  * the scrub is an allow-list of dev: an exam no list here names (ood24)
    under "exams", as a stamp or as a score path is dropped too, and D14
    stays silent; the deny-list for keys elsewhere equals driver.FINAL_EXAMS
    and report.REPORT_EXAMS without dev;
  * addresses: a JSON pointer cite and a ledger line cite resolve to their
    recorded value; a changed value, a missing pointer or a missing line
    does not check;
  * the ledger reader: 1-based physical line numbers (blank lines count), a
    partial last line skipped with a note, a bad middle line refuses the
    ledger with a note (nothing half-read);
  * metamorphic: perturbing every value under a non-dev exam key in every
    artifact (report final rows, a planted test block in state.json, the
    context) leaves the canonical bytes identical, while perturbing a dev
    value changes them (the comparison can see a change);
  * experiments sort by initialised_utc (b0_v1 before pilot_v1);
  * from_snapshot: a remote.py-shaped snapshot record (report.json without
    its final table plus derived.report_final_dev, the ledger as numbered
    entries, possibly from a prefix) gives the same artifacts and ledgers as
    load_dir on the files; its provenance is the cluster files' own sha256
    and size (the record's "files"), matching the files on disk, and an
    artifact built from derived values is marked derived, not given a file
    hash.

Run:  python3 tests/test_inc_ap_evidence.py
"""
import copy
import json
import pathlib
import shutil
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools.inc_autopilot import evidence as E  # noqa: E402

FIX = pathlib.Path(__file__).resolve().parent / "fixtures" / "inc_replay"
TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc_ap_evidence_"))
FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


def leaves(obj, ptr=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from leaves(v, ptr + "/" + E._esc(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from leaves(v, "%s/%d" % (ptr, i))
    else:
        yield ptr, obj


def perturb_non_dev(obj, unknown="ood24"):
    """Every number under a non-dev exam changed (strings too): under a known
    non-dev exam key anywhere and under every key but dev of an "exams"
    dict; each "exams" dict also gains an exam no list names (`unknown`)."""
    def go(x, under, parent=None):
        if isinstance(x, dict):
            out = {k: go(v, under or k in E.NON_DEV_EXAMS or (parent == "exams" and k != "dev"), k)
                   for k, v in x.items()}
            if parent == "exams" and unknown:
                out[unknown] = {"twelve": {"mean": 0.123456789, "n": 3}}
            return out
        if isinstance(x, list):
            return [go(v, under) for v in x]
        if under and isinstance(x, (int, float)) and not isinstance(x, bool):
            return x + 0.123 if isinstance(x, float) else x + 7
        if under and isinstance(x, str):
            return x + "-perturbed"
        return x
    return go(obj, False)


def copy_fixture(dst):
    shutil.copytree(FIX, dst)
    for p in dst.rglob("*"):
        if p.is_file():
            p.chmod(0o644)
    return dst


def test_allow_list():
    print("allow-list")
    ok = ["pilot_v1/exp.json", "pilot_v1/state.json", "pilot_v1/report.json", "pilot_v1/build_summary.json",
          "pilot_v1/ledger.jsonl", "pilot_v1/audit/label_audit.json", "realloop_v1/manifests/increments_summary.json",
          "realloop_v1/derived/state_runs.json", "step1/select_summary.json", "step1/admit_summary.json",
          "step1/increments_summary.json", "step1/relevance.json", "step1/select_clusters_by_source.json",
          "audit/pilot_v1_audit.json"]
    bad = ["pilot_v1/runs/final__base__s0/scores/test.json", "pilot_v1/scores/test.json",
           "pilot_v1/manifests/P0.jsonl", "splits/v1/LOCK.json", "splits/exp.json", "step1/exp.json",
           "step1/report.json", "../pilot_v1/exp.json", "pilot_v1/exp.json.bak", "/abs/pilot_v1/exp.json",
           "pilot_v1/audit/label_audit.md", "step1/select_clusters.csv", "step1/crops.csv", "_campaign/x.json",
           "pilot_v1/runs/x/run.json", "logs/exp.json", "audit/exp.json", "audit/pilot_v1_audit.md",
           "audit/pilot_v1_audit_boxes.csv", "audit/x/pilot_v1_audit.json", "audit/../pilot_v1_audit.json", None, 3]
    check("every artifact the contract names is allowed", all(E.allowed(n) for n in ok),
          [n for n in ok if not E.allowed(n)])
    check("score files, runs, manifests, locks, reserved dirs and escapes are refused",
          not any(E.allowed(n) for n in bad), [n for n in bad if E.allowed(n)])
    ev = E.from_texts({"pilot_v1/exp.json": "{}", "pilot_v1/runs/r/scores/test.json": '{"map50_95": 0.99}',
                       "splits/v1/LOCK.json": '{"x": 1}'}, "pilot_v1")
    check("from_texts records refused names", sorted(ev.refused) == ["pilot_v1/runs/r/scores/test.json",
                                                                      "splits/v1/LOCK.json"], ev.refused)
    check("a refused name's content never reaches the evidence", b"0.99" not in ev.canonical()
          and "pilot_v1/runs/r/scores/test.json" not in ev.artifacts)
    check("a bad experiment name is refused", raises(lambda: E.from_texts({}, "../x"), E.EvidenceError))
    check("audit/<exp>_audit.json is no experiment's directory artifact",
          E.exp_of("audit/pilot_v1_audit.json") is None and E.exp_of("pilot_v1/audit/label_audit.json") == "pilot_v1")


def test_never_opens():
    print("load_dir opens allow-listed files only")
    root = copy_fixture(TMP / "decoy")
    decoys = ["pilot_v1/runs/final__base__s0/scores/test.json", "pilot_v1/runs/final__base__s0/scores/ood22.json",
              "pilot_v1/runs/final__Tfinal__s0/scores/imageweeds.json", "splits/v1/test.jsonl",
              "step1/select_clusters.csv", "pilot_v1/manifests/P0.jsonl", "pilot_v1/scores/test.json"]
    for d in decoys:
        p = root / d
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"map50_95": 0.987654321}')
    (root / "splits" / "v1" / "exp.json").write_text("{}")         # a reserved dir is never an experiment
    opened = []
    real = E._read_file

    def spy(path):
        opened.append(str(pathlib.Path(path).relative_to(root)))
        return real(path)
    E._read_file = spy
    try:
        ev = E.load_dir(root, "pilot_v1")
    finally:
        E._read_file = real
    check("every opened file is allow-listed", all(E.allowed(n) for n in opened), opened)
    check("no decoy was opened", not set(decoys) & set(opened), set(decoys) & set(opened))
    check("the loader's touched list is what it opened", ev.touched == opened, (ev.touched, opened))
    check("a decoy value is nowhere in the evidence", b"0.987654321" not in ev.canonical())
    check("reserved dirs are not experiments", "splits" not in E.exp_dirs(root) and "step1" not in E.exp_dirs(root),
          E.exp_dirs(root))


def test_scrub():
    print("scrub")
    ev = E.load_dir(FIX, "pilot_v1")
    rep = ev.json("pilot_v1/report.json")
    check("every final row keeps only its dev exam",
          all(set(r["exams"]) == {"dev"} for r in rep["final"]), [sorted(r["exams"]) for r in rep["final"]])
    drop = ev.dropped.get("pilot_v1/report.json", [])
    want = ["/final/%d/exams/%s" % (i, x) for i in range(5) for x in ("imageweeds", "ood22", "ood23", "test")]
    check("the 20 dropped pointers are recorded", sorted(drop) == sorted(want), drop)
    check("the loader record lists them", ev.loader_record()["dropped"]["pilot_v1/report.json"] == drop)
    check("no non-dev exam survives anywhere", ev.leaks() == [], ev.leaks()[:5])
    check("the dev values survive", rep["final"][4]["exams"]["dev"]["twelve"]["n"] == 3)
    raw = {"a": {"test": 1, "dev": {"x": 2}}, "rows": [{"exam": "test", "v": 9}, {"exam": "dev", "v": 3},
                                                        "/x/scores/ood23.json", "/x/scores/dev.json", 4],
           "stamp": {"exam": "imageweeds", "v": 1}, "path": "/r/scores/test.json", "keep": "test",
           "nested": [[{"ood22": 5, "w": 6}]]}
    clean, dropped = E.scrub(raw)
    check("dict keys naming a non-dev exam are dropped", "test" not in clean["a"] and clean["a"]["dev"] == {"x": 2})
    check("a stamp of a non-dev exam in a list becomes None (position kept)",
          clean["rows"][0] is None and clean["rows"][1] == {"exam": "dev", "v": 3}, clean["rows"])
    check("a non-dev score path in a list becomes None; a dev one stays",
          clean["rows"][2] is None and clean["rows"][3] == "/x/scores/dev.json")
    check("a stamp or a score path under a dict key is dropped with its key",
          "stamp" not in clean and "path" not in clean)
    check("the word 'test' as a value is not a key and stays", clean["keep"] == "test")
    check("nested lists keep their shape", clean["nested"] == [[{"w": 6}]])
    check("every dropped address is recorded",
          sorted(dropped) == sorted(["/a/test", "/rows/0", "/rows/2", "/stamp", "/path", "/nested/0/0/ood22"]),
          dropped)
    fidelity = all(E.walk(raw, p) == v for p, v in leaves(clean) if v is not None)
    check("every pointer into the scrubbed copy addresses the raw value", fidelity)
    check("scrubbing twice changes nothing", E.scrub(clean)[0] == clean and E.leaks(clean) == [])

    print("scrub: an allow-list of dev")
    from weed_optimizer_framework.tools.inc import driver as D
    from weed_optimizer_framework.tools.inc import report as R
    check("the known non-dev exams are driver.FINAL_EXAMS and report.REPORT_EXAMS without dev",
          set(E.NON_DEV_EXAMS) == set(D.FINAL_EXAMS) - {"dev"} == set(R.REPORT_EXAMS) - {"dev"}
          and E.DECISION_EXAM == D.DECISION_EXAM == "dev", (E.NON_DEV_EXAMS, D.FINAL_EXAMS, R.REPORT_EXAMS))
    raw = {"final": [{"model": "m", "exams": {"dev": {"twelve": 0.7}, "ood24": {"twelve": 0.123456789},
                                              "test": {"twelve": 0.9}}}],
           "rows": [{"exam": "ood24", "v": 1}, {"exam": "dev", "v": 2}, "/r/scores/ood24.json", "/r/scores/dev.json"],
           "spec": {"exams": ["dev", "test", "ood24"]}, "ood24": {"not": "an exam key outside exams"}}
    clean, dropped = E.scrub(raw)
    check("an exam no list names (ood24) is dropped under 'exams', as a stamp and as a score path",
          clean["final"][0]["exams"] == {"dev": {"twelve": 0.7}} and clean["rows"] == [None, {"exam": "dev", "v": 2},
                                                                                         None, "/r/scores/dev.json"]
          and sorted(dropped) == ["/final/0/exams/ood24", "/final/0/exams/test", "/rows/0", "/rows/2"], (clean,
                                                                                                           dropped))
    check("a list of exam names (a run spec's 'exams') is names, not values, and stays",
          clean["spec"]["exams"] == ["dev", "test", "ood24"])
    check("leaks() applies the same rules", sorted(E.leaks(raw)) == sorted(dropped) and E.leaks(clean) == [])
    rep = json.loads((FIX / "pilot_v1" / "report.json").read_text())
    for r in rep["final"]:
        r["exams"]["ood24"] = {"twelve": {"mean": 0.123456789, "n": 3}}
    objs = {"pilot_v1/report.json": json.dumps(rep), "pilot_v1/exp.json": (FIX / "pilot_v1" / "exp.json").read_text()}
    ev = E.from_texts(objs, "pilot_v1")
    from weed_optimizer_framework.tools.inc_autopilot import diagnose as DG
    check("report.final[*].exams.ood24 never reaches the evidence and D14 stays silent",
          b"0.123456789" not in ev.canonical() and ev.leaks() == []
          and not DG.by_id(DG.detect(ev, only=("D14",)))["D14"]["fired"], ev.leaks()[:3])


def test_addresses():
    print("addresses")
    ev = E.load_dir(FIX, "pilot_v1")
    c = ev.cite("pilot_v1/report.json", "/agreement/full/rate")
    check("a pointer cite carries its value", c == {"artifact": "pilot_v1/report.json", "line": None,
                                                    "pointer": "/agreement/full/rate", "value": 3 / 7.0}, c)
    check("it checks", ev.check_cite(c))
    lc = ev.lcite(4, "/decision/p_recipe")
    check("a ledger cite names its 1-based line", lc["line"] == 4 and lc["artifact"] == "pilot_v1/ledger.jsonl"
          and lc["value"] == 0.0 and ev.check_cite(lc), lc)
    check("the entry on line 4 is gate/freeze/1", ev.lcite(4, "/id")["value"] == "gate/freeze/1")
    check("a changed value does not check", not ev.check_cite(dict(c, value=0.5)))
    check("a missing pointer does not check", not ev.check_cite(dict(c, pointer="/agreement/nope/rate")))
    check("a missing line does not check", not ev.check_cite(dict(lc, line=999)))
    check("a line cite on a non-ledger artifact does not check",
          not ev.check_cite(dict(lc, artifact="pilot_v1/report.json")))
    check("a cite of another experiment's ledger resolves", ev.check_cite(ev.lcite(1, "/type", exp="b0_v1")))
    check("a pointer with '/' and '~' in a key round-trips",
          E.walk({"a/b": {"c~d": 1}}, E.pointer("a/b", "c~d")) == 1)
    check("gpu_hours keys with ':' resolve", ev.get("pilot_v1/report.json", "/gpu_hours/chain:full/runs") == 42)
    check("the virtual loader record resolves", ev.get(E.LOADER, "/exp") == "pilot_v1")


def test_ledger_reader():
    print("ledger reader")
    lines = ['{"id": "a", "type": "x"}', "", '{"id": "b", "type": "x"}']
    rows, notes = E.parse_ledger("\n".join(lines) + "\n")
    check("blank lines count: line numbers are physical", [ln for ln, _ in rows] == [1, 3], rows)
    rows, notes = E.parse_ledger("\n".join(lines) + '\n{"id": "c", "ty')
    check("a partial last line is skipped with a note", [ln for ln, _ in rows] == [1, 3] and len(notes) == 1, notes)
    check("a bad middle line refuses the ledger", raises(lambda: E.parse_ledger('{"a": 1}\nnot json\n{"b": 2}\n'),
                                                          E.EvidenceError))
    ev = E.from_texts({"x_v1/ledger.jsonl": '{"a": 1}\nnot json\n{"b": 2}\n'}, "x_v1")
    check("from_texts does not half-read a bad ledger", "x_v1" not in ev.ledgers and ev.notes, ev.notes)
    ev = E.from_texts({"x_v1/ledger.jsonl": '{"id": "gate/r/1", "type": "gate", "v": 1}\n'
                                            '{"id": "gate/r/1", "type": "gate", "v": 2}\n'}, "x_v1")
    check("latest(): a later line replaces an earlier one with the same id",
          ev.latest("x_v1", "gate") == {"gate/r/1": (2, {"id": "gate/r/1", "type": "gate", "v": 2})})


def test_metamorphic():
    print("metamorphic: non-dev values cannot move the evidence")
    base = copy_fixture(TMP / "meta_a")
    pert = copy_fixture(TMP / "meta_b")
    rp = pert / "pilot_v1" / "report.json"
    rep = json.loads(rp.read_text())
    rp.write_text(json.dumps(perturb_non_dev(rep), indent=1))
    changed = sum(1 for (p1, v1), (p2, v2) in zip(leaves(rep), leaves(perturb_non_dev(rep))) if v1 != v2)
    check("the perturbation changes %d values" % changed, changed >= 80, changed)
    for root, extra in ((base, {}), (pert, {"test": {"map50_95": 0.99}, "ood22": [1, 2]})):
        st = {"exp": "pilot_v1", "generation": 3, "done": True, "blocked": {}}
        st.update(extra)
        (root / "pilot_v1" / "state.json").write_text(json.dumps(st))
    ctx_a = {"now_utc": "2026-09-27T10:00:00Z", "budget": {"envelope_su": 300, "spent_su": 20}}
    ctx_b = dict(ctx_a, imageweeds={"agnostic": 0.5})
    a = E.load_dir(base, "pilot_v1", context=ctx_a)
    b = E.load_dir(pert, "pilot_v1", context=ctx_b)
    check("the canonical bytes are identical", a.canonical() == b.canonical())
    check("the raw provenance hashes differ (they are not in the canonical bytes)",
          a.provenance["pilot_v1/report.json"]["sha256"] != b.provenance["pilot_v1/report.json"]["sha256"])
    rep2 = json.loads((base / "pilot_v1" / "report.json").read_text())
    rep2["final"][0]["exams"]["dev"]["twelve"]["mean"] += 0.001
    (pert / "pilot_v1" / "report.json").write_text(json.dumps(rep2))
    (pert / "pilot_v1" / "state.json").write_text((base / "pilot_v1" / "state.json").read_text())
    c = E.load_dir(pert, "pilot_v1", context=ctx_a)
    check("a changed dev value does change them", c.canonical() != a.canonical())


def test_order_and_snapshot():
    print("experiment order and remote snapshots")
    ev = E.load_dir(FIX, "pilot_v1")
    check("experiments sort by initialised_utc",
          ev.exps() == ["b0_v1", "pilot_v1", "pilot_v2", "base_b_v1", "pilot_v3"], ev.exps())
    raw_rep = json.loads((FIX / "pilot_v1" / "report.json").read_text())
    final_dev = [{"model": r["model"], "runs": r["runs"], "exams": {"dev": r["exams"]["dev"]}}
                 for r in raw_rep["final"]]
    lines = (FIX / "pilot_v1" / "ledger.jsonl").read_text().splitlines()
    entries = [{"line": i + 1, "entry": json.loads(x)} for i, x in enumerate(lines)]
    decision = {"artifacts": {"pilot_v1/exp.json": json.loads((FIX / "pilot_v1" / "exp.json").read_text()),
                              "pilot_v1/report.json": {k: v for k, v in raw_rep.items() if k != "final"},
                              "pilot_v1/build_summary.json":
                                  json.loads((FIX / "pilot_v1" / "build_summary.json").read_text()),
                              "_campaign/provenance/pilot_v1.json": {"x": 1}},
                "derived": {"report_final_dev": final_dev},
                "ledger": {"artifact": "pilot_v1/ledger.jsonl", "from_line": 0, "entries": entries,
                           "complete": True}}
    decision["omitted"] = {"pilot_v1/report.json": ["/final"]}
    import hashlib
    rec = {"verb": "snapshot", "ok": True, "utc": "2026-09-27T10:00:00Z", "decision": decision,
           "display_only": {"pilot_v1/report.json#/final": raw_rep["final"]},
           "files": {n: {"bytes": (FIX / n).stat().st_size, "sha256": hashlib.sha256((FIX / n).read_bytes()).hexdigest(),
                         "shipped": True}
                     for n in ("pilot_v1/exp.json", "pilot_v1/report.json", "pilot_v1/build_summary.json",
                               "pilot_v1/ledger.jsonl")}}
    snap = E.from_snapshot(rec, "pilot_v1")
    files = E.load_dir(FIX, "pilot_v1", exps=["pilot_v1"])
    check("live provenance: the cluster files' own sha256 and size, equal to the files' (load_dir)",
          all(snap.provenance[n]["sha256"] == files.provenance[n]["sha256"]
              and snap.provenance[n]["bytes"] == files.provenance[n]["bytes"]
              and snap.provenance[n]["source"] == "cluster file"
              for n in ("pilot_v1/exp.json", "pilot_v1/report.json", "pilot_v1/build_summary.json",
                        "pilot_v1/ledger.jsonl")),
          {n: (snap.provenance.get(n), files.provenance.get(n)) for n in ("pilot_v1/report.json",)})
    check("... report.json says it was shipped without its final table, put back from the dev column",
          "final" in snap.provenance["pilot_v1/report.json"]["shipped_as"]
          and "report_final_dev" in snap.provenance["pilot_v1/report.json"]["final_from"])
    def no_production(arts):
        # remote.py's report_final_dev rows carry model, runs and the dev exam, not 'production'
        out = copy.deepcopy(arts)
        for r in out["pilot_v1/report.json"]["final"]:
            r.pop("production", None)
        return out
    check("the same artifacts as the files (final rows: model, runs, dev)",
          no_production(snap.artifacts) == no_production(files.artifacts),
          sorted(set(snap.artifacts) ^ set(files.artifacts)))
    check("the final table's dev values sit at the files' pointers",
          all(snap.get("pilot_v1/report.json", "/final/%d/exams/dev" % i)
              == files.get("pilot_v1/report.json", "/final/%d/exams/dev" % i) for i in range(5)))
    check("the same ledger lines", snap.ledgers == files.ledgers)
    check("display_only is never read (no non-dev exam in the evidence)",
          snap.leaks() == [] and all(set(r["exams"]) == {"dev"}
                                     for r in snap.json("pilot_v1/report.json")["final"]))
    check("the provenance record is refused, not loaded", "_campaign/provenance/pilot_v1.json" in snap.refused
          and "_campaign/provenance/pilot_v1.json" not in snap.touched)
    from weed_optimizer_framework.tools.inc_autopilot import diagnose as DG
    check("a live snapshot does not trip D14 (a refused name is not a read)",
          not DG.by_id(DG.detect(snap, only=("D14",)))["D14"]["fired"])
    tail = dict(decision, ledger={"artifact": "pilot_v1/ledger.jsonl", "from_line": 20, "entries": entries[20:],
                                  "complete": True})
    prefix = {"pilot_v1": [(e["line"], e["entry"]) for e in entries[:20]]}
    snap2 = E.from_snapshot(dict(rec, decision=tail), "pilot_v1", ledger_prefix=prefix)
    check("a ledger shipped from line 21 plus the lab's prefix is the whole ledger", snap2.ledgers == files.ledgers)
    camp = {"verb": "campaign-snapshot", "experiments": {"pilot_v1": {"snapshot": rec}},
            "step1": {"verb": "step1", "decision": {"artifacts": {"step1/select_summary.json": {"sizes": {}}},
                                                    "derived": {"select_clusters_by_source": {"sources": {}}}}}}
    snap3 = E.from_snapshot(camp, "pilot_v1")
    check("a campaign-snapshot carries Step 1 and the per-source aggregate",
          "step1/select_summary.json" in snap3.artifacts and "step1/select_clusters_by_source.json" in snap3.artifacts)
    pv = snap3.provenance
    check("a derived artifact is marked derived with the hash of the JSON built, never a file hash",
          pv["step1/select_clusters_by_source.json"]["source"] == "derived"
          and "sha256" not in pv["step1/select_clusters_by_source.json"]
          and len(pv["step1/select_clusters_by_source.json"]["json_sha256"]) == 64
          and pv["step1/select_summary.json"]["source"] == "derived"
          and "no cluster file hash" in pv["step1/select_summary.json"]["note"], pv)
    check("the campaign snapshot keeps each experiment's cluster file hashes",
          pv["pilot_v1/exp.json"]["sha256"] == files.provenance["pilot_v1/exp.json"]["sha256"])
    check("a record that is not a snapshot is refused", raises(lambda: E.from_snapshot({"verb": "status"}, "x"),
                                                               E.EvidenceError))


def main():
    try:
        test_allow_list()
        test_never_opens()
        test_scrub()
        test_addresses()
        test_ledger_reader()
        test_metamorphic()
        test_order_and_snapshot()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
