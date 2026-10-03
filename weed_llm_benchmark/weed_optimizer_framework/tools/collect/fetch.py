"""Lever L16: fetch one source into content-addressed staging
(docs/CONTINUOUS_LOOP.md §3.1, §6.3 L16, §6.6 limits, §7.2, §7.5).

    collect fetch --source ID [--max-bytes B] [--candidates PATH]

Order of work, under the intake lock (one writer):
  1. the candidate: from the candidates file (--candidates, else the one the
     last plan recorded in intake/plan_latest.json), else the known item of
     that id (a known item stays fetchable after a discovery miss);
  2. the pre-checks that need no network (never-train, quarantine, a source
     registered outside intake, a closed source, attempts, ownership by
     another campaign, the placement of the provider when inside Slurm);
  3. describe (a fresh licence, class list and size), the prefilter, and every
     pre-check again (prefilter.precheck): a failure appends a held or closed
     event with its codes and raises a Refusal whose risk says whether a
     person is asked (R3);
  4. the files: the provider's listing, the known item's select block, the
     default data and skip patterns; ordered index files first;
  5. the byte plan: at most min(--max-bytes, the per-source cap minus what was
     fetched, the daily cap minus today's bytes, the envelope minus the total,
     the free disk minus the margin); files are taken in order while the
     listed sizes fit (a file of unknown size is streamed under the cap), the
     rest recorded as remaining (the next fetch continues: shards);
  6. every file is streamed to staging/<source>/blobs/<sha256> (checked
     against the size and checksum the provider publishes; a stream that
     breaks off resumes, transport.Net.download), and fetch.json lists
     name -> sha256, bytes, checksums, the redacted URL and the resumes;
  7. a fetched event (bytes, seconds, job id), or fetch_failed.
A file already in staging with its recorded sha256 is not fetched again.
"""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

from . import (ByteCapExceeded, ChecksumMismatch, CollectError, CredentialsMissing, FORMATS, ProviderError,
               Refusal, StaleInput, header, intake_dir, intake_lock, now_utc, read_json, safe_name, sha256_file,
               staging_dir, write_json_atomic)
from . import licence as LIC
from . import prefilter as PF
from . import providers as P
from . import state as S
from .transport import Net, redact

CAND_POINTER = "plan_latest.json"
STATIC_CODES = ("never_train", "quarantined", "registered_outside_intake", "closed", "attempts_exhausted",
                "not_placed_on_cluster", "provider_not_configured")


def load_candidates(inc=None, path=None):
    """The candidates list of a candidates file, or of the one the last plan
    recorded (intake/plan_latest.json, checked against the sha256 it
    records: a changed file is a stale input and refuses), else []. A
    pointer whose file is absent here (a plan run on another machine) gives
    [], so only the known items are fetchable, and missing_candidates()
    says why."""
    p = path
    want = None
    if p is None:
        ptr = intake_dir(inc) / CAND_POINTER
        if ptr.is_file():
            pd = read_json(ptr, "plan pointer")
            p, want = pd.get("path"), pd.get("sha256")
    if not p or not os.path.isfile(str(p)):
        return []
    if want is not None and sha256_file(p) != want:
        raise StaleInput("the candidates file %s no longer hashes to what plan_latest.json records (%s): run "
                         "plan again (lever L15)" % (p, str(want)[:12]))
    doc = read_json(p, "candidates")
    if doc.get("format") != FORMATS["candidates"]:
        raise CollectError("%s is not a %s" % (p, FORMATS["candidates"]))
    return doc.get("candidates") or []


def missing_candidates(inc=None, path=None):
    """Why no candidates file was read ("" when one was): the detail a refusal
    of an unknown source carries, so a pointer to another machine's file is
    never a silent miss."""
    if path is not None:
        return "" if os.path.isfile(str(path)) else "the candidates file %s does not exist here" % path
    ptr = intake_dir(inc) / CAND_POINTER
    if not ptr.is_file():
        return "no plan has run here (no %s)" % ptr
    p = read_json(ptr, "plan pointer").get("path")
    if not p or not os.path.isfile(str(p)):
        return "%s points at %s, which does not exist on this machine (a plan run elsewhere is not synced)" % (ptr, p)
    return ""


def find_candidate(cfg, source_id, candidates, net=None):
    """The candidate record of a source id: the candidates file's, else the
    known item's (by its id or source id)."""
    for c in candidates or []:
        if c.get("source_id") == source_id:
            return dict(c)
    it = cfg.known_item(source_id)
    if it is not None and it.get("provider") and it.get("ref"):
        return {"source_id": it["id"], "provider": it["provider"], "ref": it["ref"], "title": it.get("title"),
                "known_item": True, "known_item_id": it["id"], "known_item_name": it.get("name"), "found_by": []}
    return None


def placement_doc(inc=None):
    p = intake_dir(inc) / "placement.json"
    return read_json(p, "placement") if p.is_file() else None


def context(cfg, inc=None, rows=None, net=None, now=None, never_train=None, providers=None):
    """What the pre-checks read: the folded state, the registry, the
    never-train slugs, the credentials, the copy-scan state, today's and all
    bytes, the placement."""
    import datetime
    rows = S.read(inc) if rows is None else rows
    now = now or now_utc()
    creds = {}
    for name in (providers or cfg.providers(enabled_only=True)):
        try:
            creds[name] = P.get(cfg, name, net or Net(cfg.raw["download"])).credentials()
        except CollectError as e:
            creds[name] = (False, str(e))
    return {"state": S.fold(rows), "registry": PF.load_registry(), "never_train": never_train if never_train
            is not None else PF.never_train_slugs(), "creds": creds, "copy_scan": PF.copy_scan_ready(cfg, inc),
            "bytes_today": S.bytes_since(rows, now - datetime.timedelta(hours=24)), "bytes_total": S.bytes_total(rows),
            "placement": placement_doc(inc), "now": now}


def _record_refusal(inc, source, cand, check, what="fetch"):
    fs = check["failures"]
    worst = max(fs, key=lambda x: PF.ACTION_RANK[x["action"]])
    ev = "closed" if worst["action"] == "close" else "held"
    if worst["action"] == "refuse":
        return
    S.append(inc, source, ev, provider=cand.get("provider"), ref=cand.get("ref"), reason=worst["code"],
             codes=[x["code"] for x in fs], risk=check["risk"], stage=what,
             detail="; ".join("%s: %s" % (x["code"], x["detail"]) for x in fs)[:2000])


def _merge(cand, meta):
    """The candidate refreshed by a live describe: the provider's fields win,
    the plan's found_by stays, and every field the prefilter derives (the
    licence verdict, the lab group, the holds, provenance clearance, the
    decision: prefilter.DERIVED) is dropped, so a stale candidates file never
    decides a fetch: decide() derives them again from the fresh record."""
    out = {k: v for k, v in cand.items() if k not in PF.DERIVED}
    for k, v in meta.items():
        if v is not None and v != [] or k not in out:
            out[k] = v
    out["found_by"] = cand.get("found_by") or meta.get("found_by") or []
    return out


def _is_target_code_fn(cfg, names):
    targets = {t["taxon"].lower(): t for t in cfg.targets if t.get("taxon")}
    genera = {t["taxon"].lower() for t in cfg.targets if t.get("rank") == "genus"}

    def is_target(code):
        b = cfg.eppo_binomial(code)
        if not b:
            return False
        if names is not None:
            st = names.status(b)
            if st is not None:
                return st.get("status") in ("target", "target_synonym") and bool(st.get("target"))
        bl = b.lower()
        return bl in targets or bl.split(" ")[0] in genera
    return is_target


def fetch(cfg, source_id, max_bytes=None, candidates_path=None, net=None, inc=None, now=None, testing=False,
          names=None, targets=None):
    """L16 (module docstring). Returns the result record; a refusal raises
    Refusal after recording it."""
    from .targets import Targets
    net = net or Net(cfg.raw["download"])
    t0 = time.time()
    with intake_lock(inc, what="collect fetch %s" % source_id):
        rows = S.read(inc)
        cands = load_candidates(inc, candidates_path)
        cand = find_candidate(cfg, source_id, cands, net)
        if cand is None:
            why = missing_candidates(inc, candidates_path)
            raise Refusal("unknown_source", "%s is neither in the candidates file nor a known item%s"
                          % (source_id, (" (%s)" % why) if why else ""), action="refuse")
        source_id = cand["source_id"]
        ctx = context(cfg, inc, rows, net=net, now=now, providers=[cand["provider"]]
                      if cand.get("provider") in cfg.providers() else [])
        ctx["max_bytes"] = max_bytes
        # 2. the pre-checks that need no network
        pre0 = PF.precheck(dict(cand, decision={"status": PF.KEPT}, licence={"class": "permissive"},
                                annotation="boxes"), cfg, ctx)
        static = [f for f in pre0["failures"] if f["code"] in STATIC_CODES or f["code"].startswith("owned_by_")
                  or f["code"] == "not_fetchable"]
        if static:
            chk = {"ok": False, "risk": "R3" if any(f.get("risk") == "R3" for f in static) else None,
                   "failures": static}
            _record_refusal(inc, source_id, cand, chk)
            raise PF.refusal_from(chk)
        prov = P.get(cfg, cand["provider"], net)
        # 3. describe, prefilter, every pre-check
        try:
            meta = prov.describe(cand["ref"])
        except CredentialsMissing as e:
            chk = {"ok": False, "risk": "R3", "failures": [{"code": "credentials_missing", "detail": str(e),
                                                             "action": "hold", "risk": "R3"}]}
            _record_refusal(inc, source_id, cand, chk)
            raise PF.refusal_from(chk)
        except ProviderError as e:
            S.append(inc, source_id, "fetch_started", provider=cand["provider"], ref=cand["ref"])
            S.append(inc, source_id, "fetch_failed", provider=cand["provider"], ref=cand["ref"],
                     reason="describe_failed", detail=str(e)[:500], seconds=round(time.time() - t0, 3))
            raise Refusal("describe_failed", str(e), action="refuse")
        meta["source_id"] = source_id
        names = names if names is not None else _names_or_none(cfg, inc)
        targets = targets or Targets(cfg, names)
        dec = PF.decide(_merge(cand, meta), cfg, names or _EmptyNames(cfg), targets,
                        never_train=ctx["never_train"])
        LIC.refresh(dec, cfg.raw["licence_policy"])
        chk = PF.precheck(dec, cfg, ctx)
        if not chk["ok"]:
            _record_refusal(inc, source_id, dec, chk)
            raise PF.refusal_from(chk)
        # 4. files
        ki = cfg.known_item(dec.get("known_item_id")) if dec.get("known_item_id") else None
        select = dict((ki or {}).get("select") or {})
        select["_is_target_code"] = _is_target_code_fn(cfg, names)
        S.append(inc, source_id, "fetch_started", provider=dec["provider"], ref=dec["ref"])
        try:
            listing = prov.files(meta, select)
        except (ProviderError, CredentialsMissing) as e:
            S.append(inc, source_id, "fetch_failed", provider=dec["provider"], ref=dec["ref"],
                     reason="listing_failed", detail=str(e)[:500], seconds=round(time.time() - t0, 3))
            raise Refusal("listing_failed", str(e), action="refuse")
        chosen = P.select_files(listing, select)
        if not chosen:
            S.append(inc, source_id, "fetch_failed", provider=dec["provider"], ref=dec["ref"],
                     reason="no_data_files", detail="no file of %d matches the selection" % len(listing),
                     seconds=round(time.time() - t0, 3))
            raise Refusal("no_data_files", "no file of the %d listed matches the selection" % len(listing),
                          action="refuse")
        # 5. the byte plan
        sdir = staging_dir(source_id, inc)
        prev = _read_fetch(sdir)
        have = {f["name"]: f for f in (prev or {}).get("files") or [] if (sdir / "blobs" / f["sha256"]).is_file()}
        cap = _byte_cap(cfg, ctx, source_id, max_bytes, sdir)
        todo, planned, remaining = [], 0, []
        for fs in chosen:
            if fs["name"] in have:
                continue
            size = fs.get("size")
            if size is None:
                if not todo:
                    todo.append(fs)
                    planned = cap
                else:
                    remaining.append(fs["name"])
                continue
            if planned + int(size) <= cap:
                todo.append(fs)
                planned += int(size)
            else:
                remaining.append(fs["name"])
        if not todo and not have:
            chk = {"ok": False, "risk": "R3", "failures": [{"code": "over_byte_plan", "action": "hold", "risk": "R3",
                                                             "detail": "no selected file fits in %.2f GB (the caps; "
                                                             "a person may approve more)" % (cap / 1e9)}]}
            _record_refusal(inc, source_id, dec, chk)
            raise PF.refusal_from(chk)
        # 6. download
        got = dict(have)
        fetched_bytes = 0
        err = None
        for fs in todo:
            tmp = sdir / "blobs" / (".incoming_%s" % safe_name(fs["name"]))
            left = cap - fetched_bytes
            try:
                rec = prov.fetch_file(fs, tmp, max_bytes=int(left) if left >= 0 else 0, select=select)
            except ByteCapExceeded as e:
                err = ("over_byte_plan", str(e), "hold", "R3")
                break
            except (ChecksumMismatch, ProviderError, CredentialsMissing, OSError) as e:
                err = ("download_failed", str(e), "refuse", None)
                break
            sha = rec["sha256"]
            final = sdir / "blobs" / sha
            if final.exists():
                os.unlink(tmp)
            else:
                os.replace(tmp, final)
            fetched_bytes += int(rec["bytes"])
            got[fs["name"]] = {"name": fs["name"], "sha256": sha, "bytes": int(rec["bytes"]),
                               "checksums": {k: v for k, v in rec.items() if k not in ("bytes", "resumes")},
                               "published": fs.get("checksums") or {}, "role": fs.get("role"),
                               "group": fs.get("group"), "url": redact(fs["url"]) if fs.get("url") else
                               "ftp:%s" % fs.get("ftp_path"), "fetched_utc": _utc(),
                               "resumes": int(rec.get("resumes") or 0)}     # a stream that broke off and resumed
        names_left = [f["name"] for f in chosen if f["name"] not in got]
        doc = header("fetch", cfg, testing=testing)
        doc.update({"source_id": source_id, "provider": dec["provider"], "ref": dec["ref"],
                    "version": dec.get("version"), "title": dec.get("title"), "url": dec.get("url"),
                    "known_item": dec.get("known_item_id"), "licence": dec["licence"],
                    "lab_group": dec.get("lab_group"), "lab_group_basis": dec.get("lab_group_basis"),
                    "evaluation_lab": dec.get("evaluation_lab"), "provenance_cleared": dec.get("provenance_cleared"),
                    "hold_until": dec.get("hold_until"), "exhaustive_labels": dec.get("exhaustive_labels"),
                    "classes": dec.get("classes"), "class_status": dec.get("class_status"),
                    "decision": dec["decision"], "estimate": dec.get("estimate"), "annotation": dec.get("annotation"),
                    "files": [got[k] for k in sorted(got)], "listed": len(listing), "selected": len(chosen),
                    "remaining": names_left, "complete": not names_left,
                    "bytes": sum(f["bytes"] for f in got.values()), "select": {k: v for k, v in select.items()
                                                                               if not k.startswith("_")},
                    "attempt": (ctx["state"].get(source_id) or {}).get("attempts", 0) + 1})
        if got:
            write_json_atomic(sdir / "fetch.json", doc)
        secs = round(time.time() - t0, 3)
        if err is not None:
            S.append(inc, source_id, "fetch_failed", provider=dec["provider"], ref=dec["ref"], reason=err[0],
                     detail=err[1][:500], seconds=secs, bytes_partial=fetched_bytes)
            if fetched_bytes:
                S.append(inc, source_id, "fetched", provider=dec["provider"], ref=dec["ref"], bytes=fetched_bytes,
                         seconds=0, complete=False, files=len(got))
            raise Refusal(err[0], err[1], action=err[2], risk=err[3])
        S.append(inc, source_id, "fetched", provider=dec["provider"], ref=dec["ref"], bytes=fetched_bytes,
                 seconds=secs, complete=not names_left, files=len(got), remaining=len(names_left))
    return {"status": "fetched", "source": source_id, "bytes": fetched_bytes, "files": len(got),
            "complete": not names_left, "remaining": len(names_left), "staging": str(sdir),
            "fetch_sha256": sha256_file(sdir / "fetch.json") if (sdir / "fetch.json").exists() else None}


def _utc():
    from . import utc
    return utc()


def _read_fetch(sdir):
    p = sdir / "fetch.json"
    return read_json(p, "fetch record") if p.is_file() else None


def _byte_cap(cfg, ctx, source_id, max_bytes, sdir):
    bu = cfg.budgets()
    st = (ctx.get("state") or {}).get(source_id) or {}
    per = float((bu.get("approved_bytes") or {}).get(source_id) or bu["bytes_per_source"]) - float(st.get("bytes") or 0)
    daily = float(bu["bytes_daily"]) - float(ctx.get("bytes_today") or 0)
    env = float(bu["bytes_envelope"]) - float(ctx.get("bytes_total") or 0)
    caps = [per, daily, env]
    if max_bytes is not None:
        caps.append(float(max_bytes))
    free = disk_free(sdir)
    if free is not None:
        caps.append(float(free) - float(bu.get("disk_margin_bytes", 20e9)))
    return max(0, int(min(caps)))


def disk_free(path):
    """Free bytes of the filesystem holding path (a local safeguard; the
    project quota on /ocean is read by the autopilot's snapshot, D27)."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(str(path)).free
    except OSError:
        return None


def _names_or_none(cfg, inc):
    from . import names as NM
    try:
        return NM.load(cfg, inc)
    except CollectError:
        return None


class _EmptyNames(object):
    """A resolver stand-in with no cache: every lookup is a miss (pending)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.provenance = {"funnel_cache": None, "names_layer": None}
        self.misses = []

    def status(self, name, joined_target=None):
        if joined_target:
            return {"status": "target", "via": "join", "taxon": None, "rank": None, "target": joined_target,
                    "ambiguous": False}
        self.misses.append(name)
        return None
