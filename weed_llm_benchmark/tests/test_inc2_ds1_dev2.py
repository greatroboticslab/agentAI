#!/usr/bin/env python3
"""DS1 deviation 2 (docs/CONTINUOUS_LOOP.md, "Deviation (2026-10-30): DS1 deviation 2";
docs/prereg/DS1_dev2_2026-10-30.md).

Fixtures: tests/fixtures/ds1/dev2/, read only from the cluster on 2026-10-30, every file checked against its sha256
(MANIFEST.json), the images and labels also against the cluster's sha256s.

- item 1, image sizes: Step 1's v1-pool rows carry no image size, so the pools step's source rule (base3's) saw no
  weed box and dropped every row of rf_tuf__weed-3434e and rf_weed-detection-o9sfb__weed-detection-in-a-field as
  source_convention (main 98ef308: 1,019 and 1,229 live rows, 0 pooled). With the screen's own candidate rows of the
  two sources and their images, the pools step now reads each size from the header and pools them; a row whose
  header cannot be read is dropped 'unreadable'; base3's rules refuse a live row without a size;
- item 2, the intake rows Step 1 keyed otherwise: 21,979 admissions of ten intake batches (a9s9v 8,946) had no file
  record under their admission key (the manifest spells '.rf.', Step 1 '_rf_'); two a9s9v admissions become
  candidates (on main 98ef308 neither);
- item 3, the superseded pool records: the six records the pools step wrote at 05:38Z (by sha256) are moved aside,
  record and rows, into pools/p1/superseded/dev2/ with a MANIFEST; nothing deleted; a record with other bytes stays;
  a cut-off move completes; a folder copy or rows with other bytes refuse; the status reads them as not pooled until
  then, and the pools step moves them first and pools their sources again;
- item 4, SIU: a queued row's holds are the queue's (SIU's 123,348 queue rows hold none; the screen read the intake
  record's h6_scan, released since by serve-holds); a row the queue does not hold keeps its intake record's; DS1
  reads SIU past its stream quarantine when every cite is D31, never under D28 or with the cites unknown; SIU's rows
  that screen p1_r001 held out are screened again once, the new record replacing the old (no counts summed), its
  candidates replacing the old ones; the plan proposes that screen and holds S1 until it is taken;
- item 5, the reopen: data:1:screen and data:1:pools, as the lab's state holds them, are reopened once under the
  deviation's marker once the deployed tree holds its file, their spend kept (1.0 + 0.9517 and 2.0983 GPU-h); the
  earlier blocks (failed items only) do not reopen a missing item; the gate holds every DS1 lever until the deployed
  tree holds the file;
- the registration: the deviation's byte copy hashes to its pin (inc2.ds1_data.DEVIATION_2 and the domain's
  ds1.deviations; inc2/ds1.py stays byte for byte as deviation 1 left it, since the reproduction records pin it), ends
  docs/CONTINUOUS_LOOP.md after deviation 1's section, its proofs stamp it and its stamp file, which names the
  superseded pool records by sha256; it is the second text, the first (b825aa66...ed4d) kept with its proofs and
  stamp file under docs/prereg/superseded/DS1_dev2_first_text/, each named by its sha256.

Run:  python3 tests/test_inc2_ds1_dev2.py
"""
import contextlib
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
import traceback
import types

TMP = pathlib.Path(tempfile.mkdtemp(prefix="test_ds1_dev2_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
os.environ["INC_SCORER_TESTING"] = "1"
PKG = pathlib.Path(__file__).resolve().parents[1]
GIT = PKG.parent
sys.path.insert(0, str(PKG))

import numpy as np  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1 as D  # noqa: E402
from weed_optimizer_framework.tools.inc2 import ds1_data as DD  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import diagnose_stream as DS  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import levers_stream as LS  # noqa: E402
from weed_optimizer_framework.tools.inc_autopilot import stream as ST  # noqa: E402

FAILURES = []
FX = PKG / "tests" / "fixtures" / "ds1" / "dev2"
SIU = "zenodo_15808623"
TUF = "rf_tuf__weed-3434e"
FIELD = "rf_weed-detection-o9sfb__weed-detection-in-a-field"
MARKER = "DS1 deviation 2 (2026-10-30)"
# the six pool records the pools step wrote at 2026-10-30T05:38Z (job 50028368), every row dropped as source_convention
LIVE_POOLS = {"project_agml__mh_weed16_weed_detection":
              "c1b0e2c5f63d97960e6a477596ee8b923264b9b3b815a3c2f70195da12fa713c",
              TUF: "43482abf523851690e9d36d8ad8089c96485138a20a48ec3e2e20e0f12661ecb",
              "rf_vizerion__weed-detection-5ell6": "c90d64f128003c30cba2af70c7625e0e589914bf1d9aa9b178d752565a3844e6",
              FIELD: "f76ee83fb74ebc9abda7ba97cd6838e12a1c00617a113bf89ffd4ec8fc9f9a90",
              "rf_yolo-gyfyq__weed-ieda7": "1ac035e0ae4a98628488ec7f92c9be710786f3e1bd624b9d0a0ca1e421eb11b1",
              "rf_zijian-peng__grass-weed-69hcm": "72bdb39ea7e0bc78c24f670edecfcf1de1b046fe5882a83e7f5ae1c4e06883b2"}
EMPTY = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
# inc2/ds1.py as main 98ef308 holds it (deviation 1's): the code map the reproduction records taken on 2026-10-30 pin
DS1_PY_DEV1 = "8075ea81756c448baac20a4a516396ec90c4cc0581734a5091628e82b91cc74f"
FIRST_TEXT = "b825aa66198553a7bae49efb6f0afdfa77ce46b71226e3791ad67b467ea3ed4d"   # deviation 2's first text, kept


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + ("" if cond else "  %s" % (str(detail)[:900],)))
    if not cond:
        FAILURES.append(name)


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except D.Ds1Refused as e:
        return e
    return None


@contextlib.contextmanager
def inc(name):
    prev = C.INC_DIR
    C.INC_DIR = TMP / name
    C.INC_DIR.mkdir(parents=True, exist_ok=True)
    try:
        yield C.INC_DIR
    finally:
        C.INC_DIR = prev


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def fx_json(name):
    return json.loads((FX / name).read_text())


def fx_jsonl(name):
    return [json.loads(x) for x in (FX / name).read_text().splitlines() if x.strip()]


def h64(text):
    return int(hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16], 16)


def hasher(path):
    """A stub of inc2.guard.image_hashes: distinct 64-bit hashes per file (no copies among the fixtures)."""
    h = h64(path)
    return h, [h64("%s/%d" % (path, i)) for i in range(8)]


def decisions():
    return {d["source"]: d for d in fx_jsonl("decisions.jsonl")}


def cand_rows():
    """The screen p1_r001's candidate rows of rf_tuf and in-a-field (W and H null, as the screen wrote them), their
    files in the fixture folder."""
    out = []
    for r in fx_jsonl("p1_r001_candidates.jsonl"):
        out.append(dict(r, image=str(FX / r["image"]), label=str(FX / r["label"])))
    return out


def screen_world(rows, sources, revision=1):
    """A screen revision as p1_r001 wrote it for these rows: its record (the live screen's of each source, its old
    model audit left out) and its candidates."""
    d = DD.out_dir() / "screen"
    d.mkdir(parents=True, exist_ok=True)
    rec = {s: {k: v for k, v in fx_json("p1_r001_sources.json")["sources"][s].items() if k != "audit_verdict"}
           for s in sources}
    base = d / ("p1_r%03d" % revision)
    with open(str(base) + "_candidates.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    pathlib.Path(str(base) + ".json").write_text(json.dumps({"format": DD.FMT_SCREEN, "phase": 1, "revision": revision,
                                                            "sources": rec, "inputs": {"step1_batches": []}}))


def inputs(**kw):
    base = dict(decisions=decisions(), hasher=hasher, read_orientation=lambda p: None, production=False)
    base.update(kw)
    return DD.Inputs(**base)


# ------------------------------------------------------------------ the fixtures
def t_fixtures():
    print("the fixtures: every file as the cluster held it (sha256)")
    m = fx_json("MANIFEST.json")
    bad = [f for f, h in m["files"].items() if sha(FX / f) != h]
    extra = sorted(str(p.relative_to(FX)) for p in FX.rglob("*") if p.is_file() and p.name != "MANIFEST.json"
                   and str(p.relative_to(FX)) not in m["files"])
    check("every fixture file hashes as MANIFEST.json records, and no other file is there", m["files"] and not bad
          and not extra, (bad, extra))
    rows = cand_rows()
    check("  the ten v1-pool rows' images and labels are the cluster's bytes", len(rows) == 10 and all(
        sha(r["image"]) == m["cluster"][r["key"]]["image_sha256"] and
        sha(r["label"]) == m["cluster"][r["key"]]["label_sha256"] for r in rows))
    check("  the six pool records are the cluster's (their pinned sha256s)", {
        p.name: sha(p) for p in (FX / "pools_p1").glob("*.json")} == {
        "%s.json" % DD.slug(s): h for s, h in LIVE_POOLS.items()})


# ------------------------------------------------------------------ item 1: image sizes
def t_sizes():
    print("item 1: the pools step reads each row's size from its header (the live rf_tuf and in-a-field rows)")
    rows = cand_rows()
    check("the screen wrote these rows without a size (Step 1's v1 pool holds none)", all(
        r["W"] is None and r["H"] is None for r in rows) and all(
        "width" not in r and "height" not in r for r in fx_jsonl("step1_pool.jsonl")))
    with inc("sizes"):
        screen_world(rows, [TUF, FIELD])
        doc, out = DD.pools(1, inputs())
        rec = doc["sources"]
        check("both sources pool all five rows each (main 98ef308: 0, every row dropped as source_convention)",
              rec.get(TUF, {}).get("admitted") == 5 and rec.get(FIELD, {}).get("admitted") == 5,
              {s: (v.get("admitted"), v.get("out")) for s, v in rec.items()})
        check("  their convention reads every box at its real size (640 x 640 headers)", all(
            (rec[s]["convention"] or {}).get("passes") and rec[s]["convention"]["boxes"] == n for s, n in
            ((TUF, 5), (FIELD, 11))) and all(r["W"] == 640 and r["H"] == 640 for rs in out.values() for r in rs),
              {s: rec[s].get("convention") for s in rec})
        check("  each record says how many sizes came from the header",
              rec[TUF].get("sizes_from_header") == 5 and rec[FIELD].get("sizes_from_header") == 5)
        got = [json.loads(x) for x in (DD.out_dir() / "pools" / "p1" / ("%s.jsonl" % DD.slug(TUF))).read_text()
               .splitlines()]
        check("  the pool file is written with the sizes", len(got) == 5 and all(r["W"] == 640 for r in got))
    rs = [dict(r, drop=None) for r in cand_rows()[:2]]
    rs[1]["image"] = str(TMP / "missing.jpg")
    n = DD.fill_sizes(rs)
    check("fill_sizes reads a header and drops a file it cannot read as 'unreadable'", n == 2 and rs[0]["W"] == 640
          and rs[0]["size_from"] == "header" and rs[1]["drop"] == "unreadable",
          [(r.get("W"), r.get("drop")) for r in rs])
    rules, _c = DD.load_rules()
    e = refused(DD.apply_rules, [dict(r, drop=None, boxes=[b[1:5] for b in r["boxes"]]) for r in cand_rows()[:3]],
                rules)
    check("base3's rules refuse a live row without a size (it would be skipped unseen)", e is not None and
          "no image size" in str(e), e)


# ------------------------------------------------------------------ item 3: the superseded pool records
def plant_pools(names=None):
    d = DD.out_dir() / "pools" / "p1"
    d.mkdir(parents=True, exist_ok=True)
    for s in names or LIVE_POOLS:
        shutil.copyfile(str(FX / "pools_p1" / ("%s.json" % DD.slug(s))), str(d / ("%s.json" % DD.slug(s))))
        (d / ("%s.jsonl" % DD.slug(s))).write_bytes(b"")
    (d / "mediatum_1717366.json").write_text('{"source": "mediatum_1717366"}')
    (d / "mediatum_1717366.jsonl").write_text('{"key": "m1"}\n')
    return d


def t_supersede():
    print("item 3: the six pool records written without sizes are moved aside, byte for byte, and pooled again")
    check("the deviation names exactly the six live records by sha256, their rows the empty file",
          {e["source"]: e["record_sha256"] for e in DD.SUPERSEDED_POOLS} == LIVE_POOLS and
          all(e["rows_sha256"] == EMPTY and e["deviation"] == 2 and e["phase"] == 1 for e in DD.SUPERSEDED_POOLS))
    with inc("supersede"):
        d = plant_pools()
        check("while they stand, the status reads them as not pooled (repool) and the pools step's done set leaves "
              "them out", DD.pools_superseded_due(1) == sorted(LIVE_POOLS) and
              DD.plan_state()["phases"]["1"]["repool"] == sorted(LIVE_POOLS))
        got = DD.supersede_pools(1)
        dd = d / "superseded" / "dev2"
        check("each is moved: record and rows into pools/p1/superseded/dev2/, the same bytes, the originals gone",
              sorted(x["source"] for x in got if x["status"] == "moved") == sorted(LIVE_POOLS) and all(
                  sha(dd / ("%s.json" % DD.slug(s))) == h and sha(dd / ("%s.jsonl" % DD.slug(s))) == EMPTY and
                  not (d / ("%s.json" % DD.slug(s))).exists() and not (d / ("%s.jsonl" % DD.slug(s))).exists()
                  for s, h in LIVE_POOLS.items()), got)
        man = json.loads((dd / "MANIFEST.json").read_text())
        dv = next((x for x in DD.deviations() if x["n"] == 2), {})
        check("  the MANIFEST records each move: both files' sha256s, when, why and the deviation by sha256",
              len(man["moves"]) == 6 and all(len(m["files"]) == 2 and m["deviation"]["sha256"] == dv.get("sha256")
                                             and m["why"] and m["moved_utc"] for m in man["moves"]), man["moves"][:1])
        check("  the rows move first and the record last, so a move cut off midway leaves the record standing (due)",
              all([f["path"].rsplit(".", 1)[-1] for f in m["files"]] == ["jsonl", "json"] for m in man["moves"]),
              [[f["path"] for f in m["files"]] for m in man["moves"][:1]])
        check("  another file stays: mediatum's pool is untouched; nothing is due any more",
              (d / "mediatum_1717366.jsonl").read_text() == '{"key": "m1"}\n' and DD.pools_superseded_due(1) == [])
        check("  a second call moves nothing (moved_before)", {x["status"] for x in DD.supersede_pools(1)} ==
              {"moved_before"})
        (d / ("%s.json" % DD.slug(TUF))).write_text('{"source": "%s", "pooled": "again"}' % TUF)
        (d / ("%s.jsonl" % DD.slug(TUF))).write_bytes(b"")
        st = {x["source"]: x["status"] for x in DD.supersede_pools(1)}
        check("  the record pooled again (other bytes, even with empty rows) stays", st[TUF] == "moved_before" and
              "again" in (d / ("%s.json" % DD.slug(TUF))).read_text(), st)
    with inc("supersede_cut"):
        d = plant_pools([TUF])
        dd = d / "superseded" / "dev2"
        dd.mkdir(parents=True)
        os.link(str(d / ("%s.jsonl" % DD.slug(TUF))), str(dd / ("%s.jsonl" % DD.slug(TUF))))
        os.unlink(str(d / ("%s.jsonl" % DD.slug(TUF))))
        check("a move cut off after the rows moved is still due (the record stands)", TUF in DD.pools_superseded_due(1))
        st = {x["source"]: x["status"] for x in DD.supersede_pools(1)}
        check("  and completes: the record moved too", st[TUF] == "moved" and not (d / ("%s.json" % DD.slug(TUF)))
              .exists() and sha(dd / ("%s.json" % DD.slug(TUF))) == LIVE_POOLS[TUF], st)
    with inc("supersede_bad"):
        d = plant_pools([TUF])
        dd = d / "superseded" / "dev2"
        dd.mkdir(parents=True)
        (dd / ("%s.json" % DD.slug(TUF))).write_text("other")
        e = refused(DD.supersede_pools, 1)
        check("a folder copy with other bytes refuses, and nothing moves", e is not None and
              (d / ("%s.json" % DD.slug(TUF))).is_file(), e)
        (dd / ("%s.json" % DD.slug(TUF))).unlink()
        (d / ("%s.jsonl" % DD.slug(TUF))).write_text('{"key": "x"}\n')
        e = refused(DD.supersede_pools, 1)
        check("rows at the path with other bytes than named refuse, and the record stays", e is not None and
              sha(d / ("%s.json" % DD.slug(TUF))) == LIVE_POOLS[TUF], e)
    with inc("supersede_pools"):
        plant_pools()
        rows = cand_rows()
        screen_world(rows, [TUF, FIELD])
        st = DD.plan_state()["phases"]["1"]
        check("the status reads a source whose record stands superseded as not pooled",
              st["sources"].get(TUF, {}).get("pooled") is False and st["repool"] == sorted(LIVE_POOLS),
              st["sources"].get(TUF))
        doc, out = DD.pools(1, inputs())
        moved = {x["source"] for x in doc.get("superseded") or [] if x["status"] == "moved"}
        check("the pools step moves the six first and pools rf_tuf and in-a-field again (5 rows each)",
              moved == set(LIVE_POOLS) and doc["sources"][TUF]["admitted"] == 5 and
              doc["sources"][FIELD]["admitted"] == 5,
              (moved, {s: v.get("admitted") for s, v in doc["sources"].items()}))
        rec = json.loads((DD.out_dir() / "pools" / "p1" / ("%s.json" % DD.slug(TUF))).read_text())
        check("  its new record stands at the path; the old one in superseded/dev2/", rec["rows"] == 5 and
              sha(DD.out_dir() / "pools" / "p1" / "superseded" / "dev2" / ("%s.json" % DD.slug(TUF))) ==
              LIVE_POOLS[TUF])


# ------------------------------------------------------------------ item 2: the intake rows Step 1 keyed otherwise
def t_keys():
    print("item 2: an intake row is found under the key Step 1 gave it (a9s9v's '.rf.' admitted as '_rf_')")
    from weed_optimizer_framework.tools.inc2 import step1_stream as S1
    d = fx_json("a9s9v_rows.json")
    lost = {b: v["admissions"].get("step1_spelling", 0) for b, v in d["by_batch"].items()
            if v["admissions"].get("step1_spelling")}
    check("the fixture: 21,979 admissions of ten intake batches are found only under Step 1's spelling (a9s9v 8,946)",
          sum(lost.values()) == 21979 and len(lost) == 10 and lost.get("b0014") == 8946, lost)
    imgs = sorted(p.name for p in (FX / "images").iterdir())
    with inc("keys") as root:
        L = S1.Layout(inc_dir=root)
        man_dir = root / "intake" / "i0014_rf_zig-zag-lnodr__weed-detection-a9s9v"
        man_dir.mkdir(parents=True)
        man, adm, queue = [], [], []
        for i, kind in enumerate(("queued", "not_admitted")):
            x = d["pick"][kind]
            man.append(dict(x["manifest"], image=str(FX / "images" / imgs[i]),
                            label=str(FX / "labels" / (imgs[i].rsplit(".", 1)[0] + ".txt"))))
            adm.append(x["adm"])
            if x["queue_row"]:
                queue.append(x["queue_row"])
        (man_dir / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in man))
        bd = L.batch_dir("b0014")
        bd.mkdir(parents=True)
        (bd / "batch.json").write_text(json.dumps({"kind": "intake", "input": d["ledger"][0]["input"]}))
        (bd / "admission.jsonl").write_text("".join(json.dumps(r) + "\n" for r in adm))
        L.ledger.parent.mkdir(parents=True, exist_ok=True)
        L.ledger.write_text(json.dumps(d["ledger"][0]) + "\n")
        L.queue.parent.mkdir(parents=True, exist_ok=True)
        L.queue.write_text("".join(json.dumps(r) + "\n" for r in queue))
        got = DD.load_candidates(1, {}, inc_dir=root)
        check("both admissions become candidates, under their admission keys (on main 98ef308: none)",
              sorted(r["key"] for r in got) == sorted(a["key"] for a in adm) and
              all("_rf_" in r["key"] for r in got), [r["key"][-40:] for r in got])
        q = next((r for r in got if r["admission"] == "whole"), {})
        n = next((r for r in got if r["admission"] == "not_admitted"), {})
        check("  the queued row with the queue's holds and its boxes; the not-admitted one a weed-level candidate",
              q.get("holds") == [] and q.get("boxes") and n and n.get("pool") in ("p3", "p5") and
              n.get("image") == man[1]["image"], (q.get("holds"), n.get("pool")))


# ------------------------------------------------------------------ item 4: SIU
def siu_rows():
    """SIU's fixture rows through candidate_row: 4 queued (the queue holds none), 2 not admitted (no queue row)."""
    d = fx_json("siu_rows.json")
    out = []
    for kind in ("queued", "not_admitted"):
        for x in d[kind]:
            out.append((kind, DD.candidate_row(x["key"], x["adm"], x["q"], dict(x["f"], label=str(FX / "labels" / (
                sorted(p.name for p in (FX / "labels").iterdir())[0]))))))
    return out


def t_siu_holds():
    print("item 4: a queued row's holds are the queue's (SIU: h6_scan released by serve-holds)")
    d = fx_json("siu_rows.json")
    check("the fixture: 4 queued SIU rows whose queue rows hold nothing and whose intake records hold h6_scan; 2 not "
          "admitted (no queue row) whose intake records hold h6_scan", len(d["queued"]) == 4 and all(
              x["q"] is not None and not x["q"].get("holds") and x["f"].get("holds") == ["h6_scan"]
              for x in d["queued"]) and len(d["not_admitted"]) == 2 and all(
              x["q"] is None and x["f"].get("holds") == ["h6_scan"] for x in d["not_admitted"]))
    rows = siu_rows()
    check("candidate_row: the queued rows hold nothing (main 98ef308: h6_scan, the intake record's)",
          all(r["holds"] == [] for k, r in rows if k == "queued"), [r["holds"] for k, r in rows])
    check("  a row the queue does not hold keeps its intake record's hold", all(
        r["holds"] == ["h6_scan"] for k, r in rows if k == "not_admitted"))


def t_siu_eligibility():
    print("item 4: DS1 reads SIU past its D31 stream quarantine (DS.0 item 14), never past D28 or unknown cites")
    rows = [r for k, r in siu_rows()]
    q = {SIU, "rf_uav-qnoms__uav-wqshy"}
    live = {SIU: ["D31"], "rf_uav-qnoms__uav-wqshy": ["D28"]}
    ctx = {"quarantined": q, "quarantine_cites": live}
    got = [DD.eligibility(r, ctx) for r in rows]
    check("under the live cites (SIU: D31) the queued rows are candidates; the not-admitted ones held h6_scan",
          got == [None] * 4 + ["held:h6_scan"] * 2, got)
    check("  SIU quarantined under D28 too, or with no cites known: out (quarantined_d28)", all(
        DD.eligibility(rows[0], {"quarantined": q, "quarantine_cites": c}) == "quarantined_d28" for c in (
            {SIU: ["D28", "D31"]}, {SIU: ["D28"]}, {}, {SIU: []}, {SIU: ["?"]})))
    other = dict(rows[0], source="rf_uav-qnoms__uav-wqshy")
    check("  a D28 source stays out, whatever SIU's cites", DD.eligibility(other, ctx) == "quarantined_d28")
    check("  the cites are read from every stream's quarantine fold", callable(getattr(DD, "stream_quarantine_cites",
                                                                                     None)))


def siu_screen_world():
    """p1_r001 as the cluster holds it for SIU: every row out as quarantined_d28, none live (its record), and the six
    fixture rows as its candidates (the holds main's candidate_row read)."""
    d = fx_json("siu_rows.json")
    old = []
    for kind in ("queued", "not_admitted"):
        for x in d[kind]:
            r = DD.candidate_row(x["key"], x["adm"], x["q"], dict(x["f"], label=str(FX / "labels" / sorted(
                p.name for p in (FX / "labels").iterdir())[0])))
            old.append(dict(r, holds=list(x["f"].get("holds") or [])))
    screen_world([DD._cand_out(r) for r in old], [SIU])
    return old


def t_rescreen():
    print("item 4: SIU's rows held out by the quarantine are screened again once, the new record replacing the old")
    with inc("rescreen"):
        siu_screen_world()
        rec0 = DD.screen_records(1)[SIU]
        check("the live screen's record of SIU: 187,770 rows out as quarantined_d28, none live", rec0["rows"] == 187770
              and rec0["live"] == 0 and rec0["out"] == {"quarantined_d28": 187770}, rec0)
        check("  so SIU's screen is due again (the status says so)", DD.rescreen_due(1) == [SIU] and
              DD.plan_state()["phases"]["1"]["rescreen"] == [SIU] and DD.rescreen_due(2) == [])
        fresh = [r for k, r in siu_rows()]
        inp = inputs(candidates=fresh, quarantined={SIU}, quarantine_cites={SIU: ["D31"]}, decisions={})
        doc = DD.screen(1, inp)
        check("the next screen takes SIU's rows again (all six, though p1_r001 covered them)",
              doc["candidates"] == 6 and (doc.get("rescreen") or {}).get(SIU, {}).get("of") == ["p1_r001.json"],
              {k: doc.get(k) for k in ("candidates", "rescreen", "status")})
        rec = DD.screen_records(1)[SIU]
        check("  its record replaces p1_r001's: 6 rows, 4 live, 2 held h6_scan (not 187,776 rows summed)",
              rec["rows"] == 6 and rec["live"] == 4 and rec["out"] == {"held:h6_scan": 2} and
              rec.get("rescreen_of") == ["p1_r001.json"], {k: rec.get(k) for k in ("rows", "live", "out")})
        check("  SIU's units are EPPO codes, suggested per code (its species-role table on the sheet)",
              set(rec.get("units") or {}) >= {"eppo:SORHA", "eppo:SORVU", "eppo:SETFA", "eppo:CHEAL"} and
              rec["units"]["eppo:SORVU"]["suggested"] == "crop", rec.get("units"))
        sr = DD.screened_rows(1)
        check("  each key is read once, from the new revision (its holds the queue's)", len(sr) == 6 and
              len({r["key"] for r in sr}) == 6 and all(r["holds"] == [] for r in sr if r["admission"] == "whole"),
              [(r["key"][-20:], r["holds"]) for r in sr])
        check("  and once only: not due again, a later screen takes no SIU row again",
              DD.rescreen_due(1) == [] and DD.screen(1, inp)["candidates"] == 0)
        ps = DD.plan_state()["phases"]["1"]
        check("  the status lists SIU (live, its sheet not decided) and no screen due", SIU in ps["sources"] and
              not ps["sources"][SIU]["decided"] and ps["rescreen"] == [], ps.get("rescreen"))


# ------------------------------------------------------------------ the plan
def t_plan():
    print("the plan: the screen proposed for SIU's screen taken again; the pools for the records moved aside; S1 waits")
    cfg = LS.load_domain("weed")["ds1"]

    def fs(p1, items):
        return {"status": {"phases": {"1": p1}, "md_s0": {"drafts": 1, "draft": "draft_r001", "power": True,
                                                            "frozen": True},
                           "arms": {"S0": {"images": 39000}, "KF": {"images": 20000}},
                           "sheets": {"source_batches": 0, "ln_batches": 0, "mtw_batches": 0, "rendered": [],
                                      "audit_pending": [], "audit_sets": {}}, "scores": {}, "verdicts": {},
                           "long": {"arms": {}, "pair": None}, "ra": True, "power_final": True},
                "items": items, "today": "2026-10-30", "phase1_gb": 33.3, "audit_render_for": None}
    pooled = {"decided": True, "decision": "accept_training_only", "ready": True, "pooled": True}
    p1 = {"revisions": 2, "new_batches": 0, "sources": {TUF: dict(pooled)}, "rescreen": [SIU], "repool": []}
    _o, w, c = DS.ds1_plan(cfg, fs(p1, {"data:1:screen": "missing", "data:1:pools": "missing"}))
    check("SIU's screen due: the plan proposes data:1:screen (no new batch)", [x["item"] for x in c["maint"]][:1] ==
          ["data:1:screen"], [x["item"] for x in c["maint"]])
    check("  and no manifests item for S1 while it is due", "data:1:manifests" not in [x["item"] for x in c["maint"]])
    p1 = {"revisions": 3, "new_batches": 0, "sources": {TUF: dict(pooled, pooled=False)}, "rescreen": [],
          "repool": [TUF]}
    _o, w, c = DS.ds1_plan(cfg, fs(p1, {"data:1:screen": "missing", "data:1:pools": "missing"}))
    items = [x["item"] for x in c["maint"]]
    check("a record moved aside reads not pooled: the plan proposes data:1:pools, not the screen, not S1",
          "data:1:pools" in items and "data:1:screen" not in items and "data:1:manifests" not in items, items)
    p1 = {"revisions": 3, "new_batches": 0, "sources": {TUF: dict(pooled)}, "rescreen": [], "repool": []}
    _o, w, c = DS.ds1_plan(cfg, fs(p1, {"data:1:screen": "missing", "data:1:pools": "missing"}))
    check("  with neither due and every source settled, the plan builds S1 (data:1:manifests)",
          "data:1:manifests" in [x["item"] for x in c["maint"]], [x["item"] for x in c["maint"]])


# ------------------------------------------------------------------ item 5: the reopen
def live_items():
    """data:1:screen and data:1:pools as the lab's state held them on 2026-10-30 (12:2xZ)."""
    return {"data:1:screen": {"attempts": [{"gpuh": 1.0, "how": "settled",
                                            "proposal": "13be6a115794d605d982f1270056d3c5"}],
                              "build_job_gpuh": None, "estimate": 1.0, "exp": None, "fails": 0,
                              "last_done_utc": "2026-10-29T17:47:04Z", "lever": "L30", "phase": 1, "prior_gpuh": 1.0,
                              "proposal": "d8967fc5f33dc713fb7f969b854c2014", "runs": 1, "settled": 0.9517,
                              "spend": "settled", "started_utc": "2026-10-29T16:47:04Z", "state": "missing"},
            "data:1:pools": {"build_job_gpuh": None, "estimate": 0.5, "exp": None, "fails": 0,
                             "last_done_utc": "2026-10-30T05:42:00Z", "lever": "L30", "phase": 1,
                             "proposal": "0c1bae5fd6e893309a57422b6583ff4f", "runs": 1, "settled": 2.0983,
                             "spend": "settled", "started_utc": "2026-10-30T03:32:00Z", "state": "missing"}}


def reopen_run(st, deployed):
    run = types.SimpleNamespace(st=st, dom=LS.load_domain("weed"), ev=None, utc="2026-10-30T20:00:00Z")
    run._ds1 = types.MethodType(ST.StreamRun._ds1, run)
    run.ledger, run.cards = [], []
    run._ds1_tree_file = lambda rel, sha=None, marker=None: (rel, sha) in deployed
    run._failed_ids = types.MethodType(ST.StreamRun._failed_ids, run)
    run._v3fam_fold = ST.StreamRun._v3fam_fold
    run._ledger = lambda kind, **kw: run.ledger.append((kind, kw))
    run._card = lambda kind, title, body, **kw: run.cards.append((title, body, kw))
    return run


def t_reopen():
    print("item 5: the live data:1:screen and data:1:pools reopened under the deviation, their spend kept")
    dom = LS.load_domain("weed")["ds1"]
    dv = next((x for x in DD.deviations() if x["n"] == 2), {})
    blk = next((b for b in dom.get("reopens") or [] if b.get("marker") == MARKER), {})
    check("the domain's reopen block names the two data steps, states failed and missing, the two live attempts' "
          "proposal ids, the deviation's file and sha256, and a time after both", blk.get("items") ==
          ["data:1:screen", "data:1:pools"] and sorted(blk.get("states") or []) == ["failed", "missing"] and
          sorted(blk.get("proposals") or []) == ["0c1bae5fd6e893309a57422b6583ff4f", "d8967fc5f33dc713fb7f969b854c2014"]
          and blk.get("file") == dv.get("file") and blk.get("sha256") == dv.get("sha256") and
          str(blk.get("before_utc")) > "2026-10-30T03:32:00Z", blk)
    others = {(b["file"], b["sha256"]) for b in dom["reopens"] if b.get("marker") != MARKER}
    st = {"failed_ids": [], "ds1": {"items": live_items(), "cards": {}}}
    run = reopen_run(st, others)
    check("not reopened while the deployed tree holds the earlier files only (their blocks reopen failed items only)",
          ST.StreamRun._ds1_reopen(run) == [] and st["ds1"]["items"]["data:1:pools"].get("reopened") is None)
    run = reopen_run(st, others | {(dv.get("file"), dv.get("sha256"))})
    got = ST.StreamRun._ds1_reopen(run)
    it = st["ds1"]["items"]
    check("reopened once deployed: both missing, marked with the deviation's marker, no proposal",
          sorted(got) == ["data:1:pools", "data:1:screen"] and all(
              it[k]["state"] == "missing" and it[k]["reopened"] == MARKER and it[k]["proposal"] is None
              for k in got), got)
    check("  their spend kept: the screen 1.0 + 0.9517 = 1.9517, the pools 2.0983; the attempts recorded; the "
          "proposal ids failed", abs(it["data:1:screen"]["prior_gpuh"] - 1.9517) < 1e-9 and
          abs(it["data:1:pools"]["prior_gpuh"] - 2.0983) < 1e-9 and it["data:1:pools"]["attempts"][-1] == {
              "proposal": "0c1bae5fd6e893309a57422b6583ff4f", "gpuh": 2.0983, "how": "settled"} and
          {"d8967fc5f33dc713fb7f969b854c2014", "0c1bae5fd6e893309a57422b6583ff4f"} <= set(st["failed_ids"]),
          ({k: it[k].get("prior_gpuh") for k in it}, st["failed_ids"]))
    check("  one card (L30), one ledger line", len(run.cards) == 1 and run.cards[0][2].get("lever") == "L30" and
          run.ledger[0][0] == "ds1_reopened", run.cards)
    check("  never twice under one marker", ST.StreamRun._ds1_reopen(run) == [])
    st2 = {"failed_ids": [], "ds1": {"items": {"data:1:pools": dict(live_items()["data:1:pools"],
                                                                     started_utc="2026-10-31T00:00:00Z")}}}
    check("  an attempt started after the deviation is not reopened by it", ST.StreamRun._ds1_reopen(
        reopen_run(st2, others | {(dv.get("file"), dv.get("sha256"))})) == [])
    st4 = {"failed_ids": [], "ds1": {"items": {"data:1:screen": dict(live_items()["data:1:screen"], state="failed",
                                                                     proposal="pidW")}}}
    check("  another attempt of the same item (another proposal id: a test world's, say) is not reopened by it",
          ST.StreamRun._ds1_reopen(reopen_run(st4, others | {(dv.get("file"), dv.get("sha256"))})) == [])
    st3 = {"failed_ids": [], "ds1": {"items": {"score:POWER:draft_r001": {"state": "missing",
                                                                           "started_utc": "2026-10-29T18:00:00Z"}}}}
    check("  the earlier blocks (states failed by default) do not reopen a missing item", ST.StreamRun._ds1_reopen(
        reopen_run(st3, others | {(dv.get("file"), dv.get("sha256"))})) == [])


def t_gate():
    print("the gate: no DS1 lever until the deployed tree holds the deviation's file by its pinned sha256")
    cfg = LS.load_domain("weed")["ds1"]
    files = [(cfg["amendment_file"], cfg["amendment_sha256"])] + [(r["file"], r["sha256"]) for r in
                                                                  cfg["revisions"] + cfg.get("deviations", [])]
    check("the domain pins deviation 2 (ds1.deviations) as inc2.ds1_data does (deviation 1 in inc2.ds1)", [
        (r["file"], r["sha256"]) for r in cfg.get("deviations") or []] == [(r["file"], r["sha256"]) for r in
                                                                           DD.deviations()] and
          [r["n"] for r in DD.deviations()] == [1, 2] and [r["n"] for r in D.DEVIATIONS] == [1])

    def stage(present):
        run = types.SimpleNamespace(st={"ds1": {"items": {}, "cards": {}}}, dom=LS.load_domain("weed"), ev=None,
                                    utc="2026-10-30T20:00:00Z")
        run._ds1 = types.MethodType(ST.StreamRun._ds1, run)
        run._ds1_reopen = lambda: []
        run._artifact = lambda *a, **k: {"timestamp": True}
        run._ds1_spend = lambda *a, **k: 0.0
        run._ds1_tree_file = lambda rel, sha=None, marker=None: (rel, sha) in present if sha is not None else True
        return ST.StreamRun._ds1_stage(run, {}, None)
    dv2 = next((x for x in DD.deviations() if x["n"] == 2), {})
    check("every pinned file deployed: the gate's input is true", stage(set(files))["amendment"] is True)
    check("  deviation 2's file missing: false", stage(set(files) - {(dv2.get("file"), dv2.get("sha256"))})[
        "amendment"] is False)


# ------------------------------------------------------------------ the registration
def t_registration():
    print("the registration: the deviation's byte copy, its pins, its proofs and stamp file")
    dv = next((x for x in DD.deviations() if x["n"] == 2), {})
    f = GIT / dv.get("file", "missing")
    data = f.read_bytes() if f.is_file() else b""
    check("DS1 has deviation 2; its byte copy hashes to its pin", data and hashlib.sha256(data).hexdigest() ==
          dv.get("sha256") and data.startswith(b"## Deviation (2026-10-30): DS1 deviation 2"), dv)
    check("  inc2/ds1.py is deviation 1's, byte for byte (the code the reproduction records of 2026-10-30 pin)",
          sha(pathlib.Path(D.__file__)) == DS1_PY_DEV1, sha(pathlib.Path(D.__file__)))
    cl = (GIT / "docs" / "CONTINUOUS_LOOP.md").read_bytes()
    dev1 = (GIT / D.DEVIATIONS[0]["file"]).read_bytes()
    check("  it ends docs/CONTINUOUS_LOOP.md, appended after deviation 1's section (one blank line between)",
          data and cl.endswith(dev1 + b"\n" + data))
    stamp = GIT / dv.get("stamp_file", "missing")
    text = stamp.read_text() if stamp.is_file() else ""
    check("its proofs stamp the byte copy and the stamp file (OpenTimestamps headers)",
          D.ots_digest(str(f) + ".ots") == dv.get("sha256") and text and
          D.ots_digest(str(stamp) + ".ots") == hashlib.sha256(stamp.read_bytes()).hexdigest())
    check("  the stamp file names its sha256 and commit, the amendment's, both Revisions' and deviation 1's sha256s, "
          "and the six superseded pool records' sha256s", dv.get("sha256") and dv["sha256"] in text and
          dv.get("commit") and dv["commit"] in text and D.PREREG_SHA256 in text and
          all(r["sha256"] in text for r in D.REVISIONS) and D.DEVIATIONS[0]["sha256"] in text and
          all(h in text for h in LIVE_POOLS.values()))
    blk = next((b for b in LS.load_domain("weed")["ds1"].get("reopens") or [] if b.get("file") == dv.get("file")), {})
    check("  the reopen block's before_utc is the stamp file's written_utc", blk.get("before_utc") and
          ("written_utc: %s" % blk["before_utc"]) in text, blk.get("before_utc"))
    first = GIT / "docs" / "prereg" / "superseded" / "DS1_dev2_first_text"
    kept = sorted(first.iterdir()) if first.is_dir() else []
    check("  the first text (b825aa66...ed4d, never deployed), its proofs and stamp file are kept unchanged under "
          "docs/prereg/superseded/DS1_dev2_first_text/, each named in the stamp file by its sha256",
          [p.name for p in kept] == ["DS1_dev2_2026-10-30.md", "DS1_dev2_2026-10-30.md.ots",
                                     "DS1_dev2_2026-10-30.stamp.txt", "DS1_dev2_2026-10-30.stamp.txt.ots"] and
          sha(kept[0]) == FIRST_TEXT and D.ots_digest(str(kept[0]) + ".ots") == FIRST_TEXT and
          all("%s  %s" % (sha(p), p.name) in text for p in kept) and dv.get("sha256") != FIRST_TEXT,
          [p.name for p in kept])


def main():
    for fn in (t_fixtures, t_sizes, t_keys, t_supersede, t_siu_holds, t_siu_eligibility, t_rescreen, t_plan, t_reopen,
               t_gate, t_registration):
        try:
            fn()
        except Exception as e:                   # noqa: BLE001
            check("%s ran" % fn.__name__, False, "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc()[-2500:]))
    print("\n%d failure(s), 0 skipped: %s" % (len(FAILURES), "; ".join(FAILURES[:5])))
    shutil.rmtree(str(TMP), ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
