#!/usr/bin/env python3
"""Lever L16, fetch (docs/CONTINUOUS_LOOP.md §3.1, §6.3 L16, §6.6 limits, §7.2,
§7.5), against a fake network.

Pinned:
  * a known item is fetchable without a candidates file (after a discovery
    miss): staging holds content-addressed blobs (named by their sha256),
    fetch.json lists every file with its sha256, bytes, published checksum
    and a redacted URL, the licence, the lab group and the decision;
    sources.jsonl gets fetch_started and fetched (bytes, seconds) and its
    hash chain verifies;
  * a second fetch of a complete source downloads nothing again;
  * refusals, each recorded where the state changes: an unknown source
    (refused, nothing recorded); another campaign's source (held, before any
    network call); an evaluation-lab source without the copy scan (held, R3);
    over the per-source cap (held, R3); inside Slurm a provider not placed on
    the cluster (refused before any network call), allowed once the probe
    placed it;
  * shards: a byte cap that fits only some files fetches them in order and
    records the rest as remaining (complete false); the next fetch continues;
  * a failed download is a fetch_failed event; three of them close the
    source;
  * the candidates file the last plan recorded (plan_latest.json) is used
    when --candidates is not given, checked against the sha256 it records (a
    changed file refuses); a pointer to a file this machine lacks is named
    in the unknown source's refusal;
  * a stale candidates file never decides: its licence verdict, clearance,
    holds and lab group are derived again from the fresh describe;
  * one writer: a held intake lock refuses a fetch; across nodes the owner
    file refuses where flock is not coherent (ENOSYS), and a dead writer's
    owner file (its pid gone, its Slurm job ended) is taken over;
  * a sources.jsonl whose hash chain is broken is never appended to.

Run:  python3 tests/test_collect_fetch.py
"""
import hashlib
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_collect_world as W  # noqa: E402

TMP = W.setup("collect_fetch_")
check, raises = W.check, W.raises
PAGS = "5c78d067-8750-4803-9cbe-57df8fae55e4"
PAGS_SID = "weedai_" + PAGS


def sha(b):
    return hashlib.sha256(b).hexdigest()


def pags_net(cfg, z):
    base = cfg.provider("weedai")["base_url"]
    info = {"metadata": {"name": "Palmer amaranth Growth Stage - 8 (PAGS8)",
                         "license": "https://creativecommons.org/licenses/by/4.0/", "description": "growth stages"},
            "agcontexts": [{"n_images": 2, "category_statistics": {
                "weed: amaranthus palmeri (BBCH10-12)": {"image_count": 2, "bounding_box_count": 2}}}],
            "head_version": 1}
    return W.make_net({("GET", base + "/api/upload_info/" + PAGS): (200, info),
                       ("GET", base + "/code/download/%s.zip" % PAGS): (200, z, {"Content-Length": str(len(z))})})


def events(sid=None):
    from weed_optimizer_framework.tools.collect import state as S
    return [r for r in S.read() if sid is None or r["source"] == sid]


def zen_rec(rid, files):
    return {"id": rid, "metadata": {"title": "Plant boxes %s" % rid, "description": "Palmer amaranth bounding boxes",
                                    "license": {"id": "cc-by-4.0"}},
            "files": [{"key": k, "size": len(v), "checksum": "md5:%s" % hashlib.md5(v).hexdigest(),
                       "links": {"self": "https://zenodo.org/api/records/%s/files/%s/content" % (rid, k)}}
                      for k, v in files.items()]}


def zen_net(rid, files, bad=None):
    routes = {("GET", "https://zenodo.org/api/records/%s" % rid): (200, zen_rec(rid, files))}
    for k, v in files.items():
        routes[("GET", "https://zenodo.org/api/records/%s/files/%s/content" % (rid, k))] = (200, bad if bad else v)
    return W.make_net(routes)


def write_candidates(path, cands):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"format": "collect-candidates/1", "candidates": cands}))
    return path


def test_known_item(cfg):
    from weed_optimizer_framework.tools.collect import fetch as F
    from weed_optimizer_framework.tools.collect import staging_dir, sources_ledger, verify_chain
    print("a known item, fetched")
    z = W.zip_bytes({"a.json": b"{}"})
    net = pags_net(cfg, z)
    r = F.fetch(cfg, "pags8", net=net, testing=True)
    sd = staging_dir(PAGS_SID)
    doc = json.loads((sd / "fetch.json").read_text())
    check("fetched by the known item's id, without a candidates file", r["status"] == "fetched"
          and r["source"] == PAGS_SID and r["complete"], r)
    check("the blob is content-addressed (named by its sha256)", (sd / "blobs" / sha(z)).read_bytes() == z)
    f0 = doc["files"][0]
    check("fetch.json lists the file with sha256, bytes and the redacted URL", doc["format"] == "collect-fetch/1"
          and f0["sha256"] == sha(z) and f0["bytes"] == len(z) and f0["url"].endswith("%s.zip" % PAGS), f0)
    check("fetch.json carries the licence, lab group and decision", doc["licence"]["id"] == "cc-by-4.0"
          and doc["lab_group"] == "TAMU" and doc["provenance_cleared"] and doc["decision"]["status"] == "kept")
    ev = [e["event"] for e in events(PAGS_SID)]
    check("sources.jsonl: fetch_started then fetched", ev == ["fetch_started", "fetched"], ev)
    check("... with bytes and seconds", events(PAGS_SID)[-1]["bytes"] == len(z) and "seconds" in events(PAGS_SID)[-1])
    check("the ledger's hash chain verifies", verify_chain(sources_ledger()) == [])
    n = len([x for x in net.log if x[1].endswith(".zip") and x[0] == "GET"])
    r2 = F.fetch(cfg, "pags8", net=net, testing=True)
    n2 = len([x for x in net.log if x[1].endswith(".zip") and x[0] == "GET"])
    check("a second fetch downloads nothing again", n2 == n and r2["bytes"] == 0 and r2["complete"], (n, n2, r2))


def test_refusals(cfg):
    from weed_optimizer_framework.tools.collect import Refusal
    from weed_optimizer_framework.tools.collect import fetch as F
    print("refusals")
    net = W.make_net({})
    e = raises(lambda: F.fetch(cfg, "nosuch_source", net=net), Refusal)
    check("an unknown source is refused, nothing recorded", e is not None and e.code == "unknown_source"
          and not events("nosuch_source"), e)
    e = raises(lambda: F.fetch(cfg, "mh_weed16", net=net), Refusal)
    check("another campaign's source is held before any network call",
          e is not None and e.action == "hold" and "owned_by_funnel" in [f["code"] for f in e.failures]
          and not net.log, (e, net.log))
    check("... and the hold is recorded with its codes", events("mendeley_d3n3mgjjbv_v2")[-1]["event"] == "held"
          and events("mendeley_d3n3mgjjbv_v2")[-1]["codes"] == ["owned_by_funnel"])
    os.environ["KAGGLE_API_TOKEN"] = "t"
    try:
        view = {"ref": "yuzhenlu/cottonweeddet3", "title": "CottonWeedDet3", "description": "bounding boxes",
                "licenseName": "CC BY 4.0", "totalBytes": 5e9}
        netk = W.make_net({("GET", "https://www.kaggle.com/api/v1/datasets/view/yuzhenlu/cottonweeddet3"): (200, view)})
        e = raises(lambda: F.fetch(cfg, "cottonweeddet3", net=netk), Refusal)
        check("an evaluation-lab source without the copy scan is held, R3 (P9)", e is not None and e.risk == "R3"
              and "copy_scan_pending" in [f["code"] for f in e.failures], e)
        check("... recorded as held, R3", events("kg_yuzhenlu__cottonweeddet3")[-1]["risk"] == "R3")
    finally:
        os.environ.pop("KAGGLE_API_TOKEN", None)
    big = {"source_id": "zenodo_501", "provider": "zenodo", "ref": "501", "title": "Plant boxes 501"}
    cpath = write_candidates(TMP / "lab" / "c1.json", [big])
    rec = zen_rec("501", {"a.zip": b"x"})
    rec["files"][0]["size"] = int(60e9)
    netb = W.make_net({("GET", "https://zenodo.org/api/records/501"): (200, rec)})
    e = raises(lambda: F.fetch(cfg, "zenodo_501", candidates_path=cpath, net=netb), Refusal)
    check("over the 50 GB per-source cap: held, R3", e is not None and e.risk == "R3"
          and "over_source_cap" in [f["code"] for f in e.failures], e)
    os.environ["SLURM_JOB_ID"] = "4242"
    try:
        netp = pags_net(cfg, W.zip_bytes({"b.json": b"{}"}))
        e = raises(lambda: F.fetch(cfg, "pags8", net=netp), Refusal)
        check("inside Slurm, a provider not placed on the cluster is refused before any network call",
              e is not None and e.code == "not_placed_on_cluster" and not netp.log, e)
        from weed_optimizer_framework.tools.collect import intake_dir
        (intake_dir() / "placement.json").write_text(json.dumps(
            {"in_slurm": True, "providers": {"weedai": {"placement": "cluster", "reachable": True}}}))
        r = F.fetch(cfg, "pags8", net=netp)
        check("... and allowed once the probe placed it there", r["status"] == "fetched", r)
        (intake_dir() / "placement.json").unlink()
    finally:
        os.environ.pop("SLURM_JOB_ID", None)


def test_shards_and_failures(cfg):
    from weed_optimizer_framework.tools.collect import Refusal
    from weed_optimizer_framework.tools.collect import fetch as F
    from weed_optimizer_framework.tools.collect import staging_dir
    print("shards and failures")
    files = {"a.zip": b"A" * 100, "b.zip": b"B" * 100, "c.zip": b"C" * 100}
    cand = {"source_id": "zenodo_600", "provider": "zenodo", "ref": "600", "title": "Plant boxes 600"}
    cpath = write_candidates(TMP / "lab" / "c2.json", [cand])
    r = F.fetch(cfg, "zenodo_600", max_bytes=250, candidates_path=cpath, net=zen_net("600", files))
    check("a cap that fits two of three files fetches them in order and records the rest", r["files"] == 2
          and not r["complete"] and r["remaining"] == 1, r)
    doc = json.loads((staging_dir("zenodo_600") / "fetch.json").read_text())
    check("fetch.json lists what remains", doc["remaining"] == ["c.zip"] and not doc["complete"])
    r = F.fetch(cfg, "zenodo_600", max_bytes=250, candidates_path=cpath, net=zen_net("600", files))
    check("the next fetch continues with the rest", r["complete"] and r["files"] == 3 and r["bytes"] == 100, r)
    cand2 = {"source_id": "zenodo_700", "provider": "zenodo", "ref": "700", "title": "Plant boxes 700"}
    cpath2 = write_candidates(TMP / "lab" / "c3.json", [cand2])
    for i in range(3):
        e = raises(lambda: F.fetch(cfg, "zenodo_700", candidates_path=cpath2,
                                   net=zen_net("700", {"a.zip": b"good"}, bad=b"tampered")), Refusal)
    ev = [x for x in events("zenodo_700") if x["event"] == "fetch_failed"]
    check("a failed checksum is a fetch_failed event", len(ev) == 3 and ev[0]["reason"] == "download_failed", ev[:1])
    e = raises(lambda: F.fetch(cfg, "zenodo_700", candidates_path=cpath2, net=zen_net("700", {"a.zip": b"good"})),
               Refusal)
    check("three failed attempts close the source", e is not None and e.code == "attempts_exhausted"
          and e.action == "close" and events("zenodo_700")[-1]["event"] == "closed", e)


def test_stale_candidate(cfg):
    """A candidates file is an earlier plan's view: its licence verdict, holds
    and clearance never decide a fetch; the fresh describe does."""
    from weed_optimizer_framework.tools.collect import fetch as F
    from weed_optimizer_framework.tools.collect import staging_dir
    print("a stale candidates file")
    stale = {"source_id": "zenodo_900", "provider": "zenodo", "ref": "900", "title": "Plant boxes 900",
             "licence": {"id": "cc-by-4.0", "class": "permissive", "research_only": False},
             "licence_text": "cc-by-4.0", "provenance_cleared": True, "hold_until": None, "lab_group": "TAMU",
             "decision": {"status": "kept", "reasons": []}}
    cpath = write_candidates(TMP / "lab" / "c_stale.json", [stale])
    net = zen_net("900", {"a.zip": b"nc"})
    net.routes[("GET", "https://zenodo.org/api/records/900")][1]["metadata"]["license"] = {"id": "cc-by-nc-4.0"}
    r = F.fetch(cfg, "zenodo_900", candidates_path=cpath, net=net)
    doc = json.loads((staging_dir("zenodo_900") / "fetch.json").read_text())
    check("the licence is the fresh record's (now non-commercial: research-only), not the plan's",
          r["status"] == "fetched" and doc["licence"]["id"] == "cc-by-nc-4.0" and doc["licence"]["research_only"],
          doc["licence"])
    check("... and the plan's clearance, hold and lab group are derived again (held for the copy scan)",
          doc["provenance_cleared"] is False and doc["hold_until"] == "h6_scan" and doc["lab_group"] is None,
          (doc["provenance_cleared"], doc["hold_until"], doc["lab_group"]))


def test_pointer_and_lock(cfg):
    from weed_optimizer_framework.tools.collect import LockHeld, Refusal, StaleInput, intake_dir, intake_lock
    from weed_optimizer_framework.tools.collect import fetch as F
    print("the plan's pointer; one writer")
    ptr = intake_dir() / "plan_latest.json"
    ptr.write_text(json.dumps({"format": "collect-candidates/1", "path": str(TMP / "lab" / "elsewhere.json"),
                               "sha256": "0" * 64}))
    e = raises(lambda: F.fetch(cfg, "zenodo_801", net=W.make_net({})), Refusal)
    check("a pointer to a file this machine lacks: the unknown source's refusal says so (never a silent miss)",
          e is not None and e.code == "unknown_source" and "does not exist on this machine" in e.detail, e)
    cand = {"source_id": "zenodo_800", "provider": "zenodo", "ref": "800", "title": "Plant boxes 800"}
    cpath = write_candidates(TMP / "lab" / "c4.json", [cand])
    ptr.write_text(json.dumps({"format": "collect-candidates/1", "path": str(cpath), "sha256": "f" * 64}))
    e = raises(lambda: F.fetch(cfg, "zenodo_800", net=zen_net("800", {"a.zip": b"zz"})), StaleInput)
    check("a candidates file that no longer hashes as the pointer records refuses (stale input)", e is not None, e)
    ptr.write_text(json.dumps({"format": "collect-candidates/1", "path": str(cpath),
                               "sha256": sha(cpath.read_bytes())}))
    r = F.fetch(cfg, "zenodo_800", net=zen_net("800", {"a.zip": b"zz"}))
    check("the last plan's candidates file is used when none is given", r["status"] == "fetched", r)
    from weed_optimizer_framework.tools.collect import sources_ledger
    led = sources_ledger()
    good = led.read_bytes()
    lines = good.decode().splitlines(True)
    row1 = json.loads(lines[1])
    row1["source"] = str(row1["source"]) + "_edited"
    edited = lines[:1] + [json.dumps(row1, sort_keys=True, separators=(",", ":")) + "\n"] + lines[2:]
    led.write_bytes("".join(edited).encode())
    e = raises(lambda: F.fetch(cfg, "zenodo_800", net=zen_net("800", {"a.zip": b"zz"})), StaleInput)
    check("an edited line of sources.jsonl breaks its chain, and no writer appends to it (fail closed)",
          e is not None and "broken hash chain" in str(e) and led.read_bytes() == "".join(edited).encode(), e)
    led.write_bytes(good)
    with intake_lock(what="another writer"):
        e = raises(lambda: F.fetch(cfg, "zenodo_800", net=zen_net("800", {"a.zip": b"zz"})), LockHeld)
    check("a held intake lock refuses a second writer", e is not None and "another writer" in str(e), e)


def test_owner_lock(cfg):
    """One writer across nodes: the owner file (O_EXCL) holds where flock is
    not coherent across nodes (a Lustre mount without 'flock' answers ENOSYS);
    a dead writer's owner file is taken over, a live one refuses."""
    import errno
    import fcntl
    import socket
    import time
    import weed_optimizer_framework.tools.collect as COL
    from weed_optimizer_framework.tools.collect import LockHeld, intake_dir, intake_lock
    print("the writer lock across nodes")
    owner = intake_dir() / ".lock.owner"
    real_flock = fcntl.flock

    def no_flock(fd, op):
        if op & fcntl.LOCK_UN:
            return None
        raise OSError(errno.ENOSYS, "Function not implemented")
    fcntl.flock = no_flock
    try:
        with intake_lock(what="first writer"):
            e = raises(lambda: intake_lock(what="second writer").__enter__(), LockHeld)
            check("where flock answers ENOSYS (a Lustre mount without flock) the owner file alone refuses a second "
                  "writer", e is not None and "live writer" in str(e), e)
        check("... and it is released with the lock", not owner.exists())
    finally:
        fcntl.flock = real_flock
    real_sq = COL.SQUEUE
    sq = TMP / "fake_squeue"
    try:
        owner.write_text(json.dumps({"host": "other-node", "pid": 1, "job": "4711", "t": time.time()}))
        sq.write_text("#!/bin/sh\necho RUNNING\n")
        sq.chmod(0o755)
        COL.SQUEUE = str(sq)
        e = raises(lambda: intake_lock(what="w").__enter__(), LockHeld)
        check("a writer on another node whose Slurm job still runs keeps the lock", e is not None, e)
        sq.write_text("#!/bin/sh\necho 'slurm_load_jobs error: Invalid job id specified' >&2\nexit 1\n")
        with intake_lock(what="after a killed job"):
            took = json.loads(owner.read_text())["host"] == socket.gethostname()
        check("... and once its job has ended (killed at its time limit, no release) the lock is taken over", took)
        owner.write_text(json.dumps({"host": socket.gethostname(), "pid": 999999, "t": time.time()}))
        with intake_lock(what="after a dead pid"):
            pass
        check("a dead process's owner file on this host is taken over", not owner.exists())
    finally:
        COL.SQUEUE = real_sq
        if owner.exists():
            owner.unlink()


def main():
    try:
        cfg = W.config()
        W.build_cache()
        from weed_optimizer_framework.tools.collect import fetch as F
        F.disk_free = lambda p: 10 ** 13               # the test machine's free space is not the subject here
        test_known_item(cfg)
        test_refusals(cfg)
        test_shards_and_failures(cfg)
        test_stale_candidate(cfg)
        test_pointer_and_lock(cfg)
        test_owner_lock(cfg)
    finally:
        W.cleanup()
    print("\n%d failure(s)" % len(W.FAILURES))
    return 1 if W.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
