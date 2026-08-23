"""Quarantine a registry dataset instead of deleting it (SUPERWEED_PLAN S1).

A quarantined source keeps its files and registry entry but: the merge skips it
(`skipped_quarantined` stat), and the UI lists it greyed-out. Every quarantine
carries a reason and is reversible — nothing is silently destroyed.

    python -m weed_optimizer_framework.tools.quarantine mark <slug> --reason "..."
    python -m weed_optimizer_framework.tools.quarantine unmark <slug>
    python -m weed_optimizer_framework.tools.quarantine list
"""
import argparse
import json
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REG = os.path.join(REPO, "results", "framework", "dataset_registry.json")


def _mutate(fn):
    from .registry_lock import update_registry
    return update_registry(REG, fn)


def mark(slug: str, reason: str):
    def _apply(r):
        info = r.get("datasets", {}).get(slug)
        if not isinstance(info, dict):
            raise SystemExit("unknown slug: %s" % slug)
        info["status_before_quarantine"] = info.get("status")
        info["status"] = "quarantined"
        info["quarantine_reason"] = reason
        info["quarantined_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        return r
    _mutate(_apply)
    print("quarantined %s (%s)" % (slug, reason))


def unmark(slug: str):
    def _apply(r):
        info = r.get("datasets", {}).get(slug)
        if not isinstance(info, dict):
            raise SystemExit("unknown slug: %s" % slug)
        if info.get("status") == "quarantined":
            info["status"] = info.pop("status_before_quarantine", "downloaded") or "downloaded"
        info.pop("quarantine_reason", None)
        info.pop("quarantined_at", None)
        return r
    _mutate(_apply)
    print("restored %s" % slug)


def list_quarantined():
    reg = json.load(open(REG))
    rows = [(k, v.get("quarantine_reason"), v.get("quarantined_at"))
            for k, v in reg.get("datasets", {}).items()
            if isinstance(v, dict) and v.get("status") == "quarantined"]
    if not rows:
        print("no quarantined datasets")
    for k, why, when in rows:
        print("%-52s %-40s %s" % (k[:52], str(why)[:40], when))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mark"); m.add_argument("slug"); m.add_argument("--reason", required=True)
    u = sub.add_parser("unmark"); u.add_argument("slug")
    sub.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "mark":
        mark(a.slug, a.reason)
    elif a.cmd == "unmark":
        unmark(a.slug)
    else:
        list_quarantined()
