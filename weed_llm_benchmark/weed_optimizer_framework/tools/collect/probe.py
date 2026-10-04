"""The network probe (docs/CONTINUOUS_LOOP.md §7.2; an R0 lever run as one
GPU-shared job) and the collector's state summary.

probe: every configured provider's probe URL (and FTP login, for a provider
with an FTP server) is tried from the machine it runs on. Inside a Slurm job
the result is INC_DIR/intake/placement.json: a provider is placed on the
cluster only when it answered there (status "pass") and the config does not
keep it on the lab (placement.lab_only); every other provider keeps the lab
hook. Outside Slurm the
result goes to placement_lab.json and never changes the cluster placement.

summary: INC_DIR/intake/state.json (format collect-state/1), read by the
autopilot's snapshot: every source's folded state (status, reason, holds,
attempts, bytes, seconds, jobs, batches, latest yield), the recent outcomes in
time order (yield, zero or failed: the DATA lane's three-in-a-row stop-loss
reads them, each zero listing its decision reasons), bytes in the last 24 h
and in total against the caps, the ledgers' hash-chain check, the placement
and the yield floors.
"""
from __future__ import annotations

import datetime
import socket
import time

from . import (CollectError, ProviderError, batches_ledger, header, in_slurm, intake_dir, intake_lock, now_utc,
               read_json, read_jsonl, sha256_file, slurm_job_id, sources_ledger, verify_chain,
               write_json_atomic)
from . import providers as P
from . import state as S
from .transport import Net


def probe(cfg, net=None, inc=None, testing=False, providers=None):
    net = net or Net(cfg.raw["download"])
    res = {}
    slurm = in_slurm()
    for name, sec in sorted(cfg.providers(enabled_only=True).items()):
        if providers and name not in providers:
            continue
        prov = P.get(cfg, name, net)
        try:
            r = prov.probe()
        except CollectError as e:
            r = {"reachable": False, "error": str(e)[:300]}
        ftp = sec.get("ftp") or {}
        if ftp.get("host"):
            t0 = time.time()
            try:
                ses = net.ftp(ftp["host"], ftp.get("user") or "anonymous", ftp.get("password") or "")
                ses.close()
                r["ftp"] = {"reachable": True, "seconds": round(time.time() - t0, 3)}
            except (ProviderError, OSError, EOFError) as e:
                r["ftp"] = {"reachable": False, "error": str(e)[:300]}
            except Exception as e:  # noqa: BLE001 - ftplib raises its own error classes
                r["ftp"] = {"reachable": False, "error": str(e)[:300]}
            r["reachable"] = bool(r.get("reachable")) and r["ftp"]["reachable"]
        ok = bool(r.get("reachable"))
        r["status"] = "pass" if ok else "fail"
        r["lab_only"] = cfg.lab_only(name)
        r["placement"] = "cluster" if (slurm and ok and not r["lab_only"]) else "lab"
        res[name] = r
    doc = header("placement", cfg, testing=testing)
    doc.update({"in_slurm": slurm, "node": socket.gethostname(), "slurm_job_id": slurm_job_id(), "providers": res,
                "rule": "a provider is placed on the cluster only when it answered from a compute node and the "
                        "config does not keep it on the lab; every other provider keeps the lab hook"})
    name = "placement.json" if slurm else "placement_lab.json"
    with intake_lock(inc, what="collect probe"):
        write_json_atomic(intake_dir(inc) / name, doc)
    return {"status": "probed", "file": str(intake_dir(inc) / name), "in_slurm": slurm,
            "cluster": sorted(k for k, v in res.items() if v["placement"] == "cluster"),
            "unreachable": sorted(k for k, v in res.items() if not v.get("reachable"))}


def outcomes(folded):
    """[(last_ts, source, outcome, reasons)] of sources that ended an attempt:
    outcome "yield" (an intake with target boxes), "zero" (an intake with none)
    or "failed" (a failed fetch or a closed source), in time order."""
    out = []
    for sid, s in folded.items():
        ev = s.get("last_event")
        if ev == "intaken":
            y = s.get("yield") or {}
            out.append((s["last_ts"], sid, "yield" if (y.get("target_boxes") or 0) > 0 else "zero",
                        y.get("rejected") or {}))
        elif ev in ("fetch_failed",) or s.get("status") == "closed":
            out.append((s["last_ts"], sid, "failed", {"reason": s.get("reason")}))
    out.sort()
    return out


def summary(cfg, inc=None, extra=(), testing=False, now=None):
    now = now or now_utc()
    rows = S.read(inc, extra)
    folded = S.fold(rows)
    batches = read_jsonl(batches_ledger(inc), missing_ok=True)
    per_batch = {}
    for b in batches:
        p = intake_dir(inc) / b["batch"] / "summary.json"
        if p.is_file():
            d = read_json(p, "batch summary")
            per_batch[b["batch"]] = {"source": d.get("source"), "rows": d.get("rows"), "yield": d.get("yield"),
                                     "source_leak": d.get("source_leak"), "zero_yield": d.get("zero_yield")}
    placement = None
    pp = intake_dir(inc) / "placement.json"
    if pp.is_file():
        placement = read_json(pp, "placement")
    bu = cfg.budgets()
    by_status = {}
    for s in folded.values():
        k = s["status"] or "none"
        by_status[k] = by_status.get(k, 0) + 1
    ins = {}
    for name, p in (("sources", sources_ledger(inc)), ("batches", batches_ledger(inc))):
        if p.is_file():
            ins[name] = {"path": str(p), "sha256": sha256_file(p)}
    out = header("state", cfg, inputs=ins, testing=testing)
    out.update({
        "sources": folded, "by_status": by_status, "batches": per_batch,
        "recent_outcomes": [{"ts": t, "source": s, "outcome": o, "reasons": r} for t, s, o, r in outcomes(folded)],
        "bytes": {"last_24h": S.bytes_since(rows, now - datetime.timedelta(hours=24)), "total": S.bytes_total(rows),
                  "daily_cap": bu.get("bytes_daily"), "envelope": bu["bytes_envelope"],
                  "per_source_cap": bu["bytes_per_source"]},
        "chains": {"sources.jsonl": verify_chain(sources_ledger(inc)), "batches.jsonl": verify_chain(batches_ledger(inc))},
        "placement": (placement or {}).get("providers"), "floors": cfg.yield_floors(),
    })
    with intake_lock(inc, what="collect summary"):
        write_json_atomic(intake_dir(inc) / "state.json", out)
    return {"status": "summarised", "file": str(intake_dir(inc) / "state.json"), "sources": len(folded),
            "by_status": by_status, "batches": len(per_batch), "bytes_total": out["bytes"]["total"],
            "chain_problems": sum(len(v) for v in out["chains"].values())}
