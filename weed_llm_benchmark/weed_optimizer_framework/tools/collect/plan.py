"""Lever L15: discovery (docs/CONTINUOUS_LOOP.md §3.1, §6.3 L15, §7.1-7.3).

    collect plan --config C --out PATH [--classes A,B] [--providers p,q] [--resolve-names]

For every enabled provider that searches: the queries of its template kind
(targets.Targets.queries: deficit classes first, interleaved, capped at
query_grammar.max_queries_per_provider), each answered by provider.search. A
provider without its credentials is skipped and reported (card X16); a failed
query is reported, never fatal. Candidates are merged by source id, keeping
every query that found them. The known items are added whether or not a
search found them (they stay fetchable after a recorded miss). Candidates are
described (a fresh record with licence, classes, size), at most
max_describe_per_provider per provider, known items first; then the
prefilter decides, copies are superseded by the preferred licence (P6), and
the kept candidates are ranked (prefilter.rank) and pre-checked against the
current state (prefilter.precheck: ok, the reasons, and whether a person is
asked). --resolve-names (lab) asks the taxonomy authority for the names the
caches lack and appends them to the names layer, as lever L26 does; without
it a candidate whose names are missing is pending_names. With --providers,
only those providers are searched and only their known items are added.

Recall (the audit of the search, S1b, reported as the funnel's H12): a known
item is found when a candidate that a search returned matches one of its
match rules. recall = found / known items; a miss is listed with the item and
is a discovery defect, not a stop.

Outputs: the candidates file (format collect-candidates/1) at --out,
intake/plan_latest.json (its path and sha256, read by fetch and names), and a
candidate event in sources.jsonl for each source seen for the first time.
"""
from __future__ import annotations

import time

from . import (FORMATS, CollectError, CredentialsMissing, ProviderError, header, intake_dir, intake_lock,
               sha256_file, write_json_atomic)
from . import licence as LIC
from . import names as NM
from . import prefilter as PF
from . import providers as P
from . import state as S
from .config import match_any, record_text
from .fetch import CAND_POINTER, context
from .targets import Targets
from .transport import Net


def _merge_meta(a, b):
    out = dict(a)
    for k, v in b.items():
        if k in ("found_by", "raw_sha256"):
            continue
        if v not in (None, [], {}, "") or k not in out:
            out[k] = v
    out["found_by"] = list(a.get("found_by") or [])
    out["raw_sha256"] = list(a.get("raw_sha256") or []) + [x for x in (b.get("raw_sha256") or [])
                                                            if x not in (a.get("raw_sha256") or [])]
    return out


def recall(cfg, cands):
    items = cfg.known_items()
    found, missed = [], []
    for it in items:
        hits = sorted(c["source_id"] for c in cands if c.get("found_by")
                      and match_any(it.get("match") or [], c.get("provider"), c.get("ref"), c.get("title"),
                                    record_text(c)))
        if hits:
            found.append({"id": it["id"], "name": it.get("name"), "by": hits})
        else:
            missed.append({"id": it["id"], "name": it.get("name"),
                           "why": "no search result matched its match rules (a discovery defect)"})
    n = len(items)
    return {"known_items": n, "found": found, "missed": missed, "recall": round(len(found) / n, 4) if n else None,
            "decided_by": cfg.known_decided_by(),
            "rule": "a known item is found when a candidate a search returned matches one of its match rules"}


def plan(cfg, out, classes=None, providers=None, net=None, inc=None, resolve_names=False, testing=False,
         now=None, transport=None):
    """L15 (module docstring). Returns a summary record."""
    t0 = time.time()
    net = net or Net(cfg.raw["download"])
    tr = transport or (lambda url, params: net.request(url, params))
    names = NM.load(cfg, inc, allow_network=resolve_names, transport=tr if resolve_names else None)
    targets = Targets(cfg, names)
    deficit = [c for c in (classes or []) if c] or None
    targets.ordered(deficit)                              # refuses non-targets early
    qg = cfg.raw["query_grammar"]
    enabled = cfg.providers(enabled_only=True)
    if providers:
        bad = [p for p in providers if p not in enabled]
        if bad:
            raise CollectError("--providers names providers that are not enabled: %s" % bad)
        enabled = {k: v for k, v in enabled.items() if k in providers}
    found, errors, queries = {}, {}, {}
    for pname, sec in sorted(enabled.items()):
        prov = P.get(cfg, pname, net)
        if sec.get("search") is False or not prov.searchable:
            continue
        ok, detail = prov.credentials()
        if not ok:
            errors[pname] = [{"code": "credentials_missing", "detail": detail}]
            continue
        qs = targets.queries(sec.get("template") or "text", deficit,
                             limit=int(sec.get("max_queries") or qg["max_queries_per_provider"]),
                             templates=sec.get("templates"))
        queries[pname] = [{"query": q, "target": t} for q, t in qs]
        for q, t in qs:
            try:
                metas = prov.search(q, int(sec.get("max_results") or qg["max_results_per_query"]))
            except (ProviderError, CredentialsMissing) as e:
                errors.setdefault(pname, []).append({"query": q, "error": str(e)[:300]})
                continue
            for m in metas:
                c = found.get(m["source_id"])
                if c is None:
                    c = found[m["source_id"]] = dict(m, found_by=[])
                else:
                    found[m["source_id"]] = c = _merge_meta(c, m)
                if not any(f["query"] == q and f["provider"] == pname for f in c["found_by"]):
                    c["found_by"].append({"provider": pname, "query": q, "target": t})
    # the known items, found or not
    for it in cfg.known_items():
        if not it.get("provider") or not it.get("ref") or it["provider"] not in enabled:
            continue
        prov = P.get(cfg, it["provider"], net)
        sid = it["id"]
        if sid not in found:
            found[sid] = prov.meta(it["ref"], source_id=sid, title=it.get("title"))
    # describe
    described = {}
    order = sorted(found.values(), key=lambda c: (0 if cfg.known_item_for(c["source_id"], c["provider"],
                                                                           c.get("ref"), c.get("title"),
                                                                           record_text(c)) else 1,
                                                  -len(c.get("found_by") or []), c["source_id"]))
    for c in order:
        pname = c["provider"]
        cap = int((enabled.get(pname) or cfg.provider(pname)).get("max_describe") or qg["max_describe_per_provider"])
        if described.get(pname, 0) >= cap:
            c["described"] = False
            continue
        described[pname] = described.get(pname, 0) + 1
        prov = P.get(cfg, pname, net)
        try:
            meta = prov.describe(c["ref"])
            meta["source_id"] = c["source_id"]
            found[c["source_id"]] = dict(_merge_meta(c, meta), described=True)
        except (ProviderError, CredentialsMissing) as e:
            c["described"] = False
            c["describe_error"] = str(e)[:300]
            errors.setdefault(pname, []).append({"describe": c["ref"], "error": str(e)[:300]})
    never = PF.never_train_slugs()
    decided = [PF.decide(c, cfg, names, targets, deficit=deficit, never_train=never) for c in found.values()]
    for c in decided:                     # a kept candidate whose record named no licence: the platform's lookup
        if c["decision"]["status"] == PF.KEPT:
            LIC.refresh(c, cfg.raw["licence_policy"])
    PF.supersede(decided, cfg)
    ranked = PF.rank(decided)
    with intake_lock(inc, what="collect plan"):
        rows = S.read(inc)
        ctx = context(cfg, inc, rows, net=net, now=now, never_train=never, providers=list(enabled))
        for c in ranked:
            c["precheck"] = PF.precheck(c, cfg, ctx)
            # the flat fields the autopilot's DATA lane reads (inc_autopilot/stream.py)
            c["id"] = c["source_id"]
            c["found_by_search"] = bool(c.get("found_by"))
            # tri-state: True usable, False refused (closed), None unresolved (a person decides, R3; P6): an
            # unresolved licence is not a refused one, and reading it as False would drop the source unasked
            lcls = c["licence"]["class"]
            c["licence_ok"] = True if lcls in ("permissive", "research_only") else (False if lcls == "refused"
                                                                                     else None)
            c["licence_id"], c["licence_class"] = c["licence"]["id"], lcls
            c["names_unresolved"] = bool(c.get("names_pending"))
            c["expected_target_boxes"] = c["estimate"]["target_boxes"]
            c["credentials_ok"] = bool((ctx["creds"].get(c["provider"]) or (True, None))[0])
            c["image_level"] = c.get("annotation") == "image_level"
            c["annotation_type"] = c.get("annotation")
            c["copy_scan_done"] = bool(ctx["copy_scan"][0])
        rec = recall(cfg, ranked)
        missed = {m["id"] for m in rec["missed"]}
        for c in ranked:
            # known_item: the row is on the list only as the owner's known item, no search found the item
            # (the autopilot's recall counts every other row as found); known_item_id names the item for
            # any row that is, copies or matches one
            c["known_item"] = bool(c.get("known_item_role") == "primary" and c.get("known_item_id") in missed)
        doc = header("candidates", cfg, inputs={k: v for k, v in (names.provenance or {}).items() if v},
                     testing=testing)
        counts = {}
        for c in ranked:
            counts[c["decision"]["status"]] = counts.get(c["decision"]["status"], 0) + 1
        doc.update({"classes": deficit, "providers": sorted(enabled), "queries": queries, "provider_errors": errors,
                    "candidates": ranked, "counts": counts, "recall": rec, "names_misses": sorted(set(names.misses)),
                    "seconds": round(time.time() - t0, 3)})
        if resolve_names:                      # the names resolved online join the names layer (lever L26's file)
            _f, layer = NM.cache_paths(cfg, inc)
            doc["names_layer"] = names.save_layer(layer, testing=testing)
        sha = write_json_atomic(out, doc)
        write_json_atomic(intake_dir(inc) / CAND_POINTER, {"format": FORMATS["candidates"], "path": str(out),
                                                           "sha256": sha})
        seen = S.fold(rows)
        for c in ranked:
            if c["source_id"] not in seen:
                S.append(inc, c["source_id"], "candidate", provider=c["provider"], ref=c["ref"],
                         decision=c["decision"]["status"], rank=c.get("rank"))
    return {"status": "planned", "out": str(out), "sha256": sha256_file(out), "candidates": len(ranked),
            "counts": counts, "recall": rec["recall"], "missed": [m["id"] for m in rec["missed"]],
            "provider_errors": {k: len(v) for k, v in errors.items()},
            "top": [c["source_id"] for c in ranked if c.get("rank")][:10]}
