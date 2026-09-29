"""The pre-download filter and the fetch pre-checks (docs/CONTINUOUS_LOOP.md
§3.1 "Refuses when", §6.3 L16, §7.2).

decide(candidate) judges a candidate from what its provider declares before any
download: the class list (a class-name index, a class-label feature, a card),
or, for providers that declare none, the record's text. A candidate is kept
only if
  (i)   at least one declared class maps to a target through the offline
        resolver (classmap.map_class: status target or target_synonym);
  (ii)  it is a known item (the owner's list in the config); or
  (iii) the config holds a card class table for it (owner-pinned);
or, for a provider that declares no classes, when its text names a target
and shows boxes (a box word or a detection task tag).
The reject list (prefilter.reject_tokens) applies to whole tokens of class
names, so a short word never rejects a longer name that starts with it; it
rejects only a candidate with no target class. A class list holding at least
copy_candidate_min names equal to the legacy labels of the reference dataset
(prefilter.legacy_labels, the rule of the v1 old join) is a copy candidate:
recorded, never downloaded. A re-upload platform's candidate that declares one
of presumed_derivative.declares_any, or at least min_targets targets, is
presumed to derive from the evaluation lab (§7.2 [review]): lab group set,
fetched, and every row held until the embedding copy scan (h6_scan).
Candidates are ranked by expected target boxes per GB, with a bonus per
deficit class they declare (D20); ties by target classes per GB.

precheck(candidate, context) lists every reason L16 may not fetch it now, each
with the action it implies (hold, refuse, close) and the governance class of
the person's decision it asks for (R3), so the autopilot files the right item.

The never-train slugs are read from the trainer's source file by parsing it
(never imported: the collector imports no trainer).
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

from . import TOOLS_DIR, ConfigError, Refusal, in_slurm, read_json
from . import classmap as CM
from .config import record_text
from . import licence as LIC

TRAINER_SOURCE = "mega_trainer.py"          # parsed, never imported
NEVER_TRAIN_NAME = "NEVER_TRAIN_SLUGS"
INTAKE_ANNOTATION = "intake_v1"
KEPT, REJECTED, COPY, PENDING, HELD = "kept", "rejected", "copy_candidate", "pending_names", "held"
ACTION_RANK = {"close": 3, "refuse": 2, "hold": 1}


# ------------------------------------------------------------------ never-train
def never_train_slugs(path=None):
    """The never-train slug set of the trainer module, read by parsing its
    source (ast.literal_eval of the assignment). Refuses (fail closed) when
    the assignment cannot be found or read."""
    p = Path(path) if path else TOOLS_DIR / TRAINER_SOURCE
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as e:
        raise ConfigError("cannot read the never-train slugs from %s (%s): every fetch is refused" % (p, e))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == NEVER_TRAIN_NAME for t in node.targets):
            try:
                return frozenset(str(s) for s in ast.literal_eval(node.value))
            except ValueError as e:
                raise ConfigError("%s.%s is not a literal (%s)" % (p.name, NEVER_TRAIN_NAME, e))
    raise ConfigError("%s holds no %s assignment: every fetch is refused" % (p, NEVER_TRAIN_NAME))


def registry_path():
    from ..inc import common as C
    return Path(os.environ.get("COLLECT_REGISTRY") or (C.REPO / "results" / "framework" / "dataset_registry.json"))


def load_registry(path=None):
    """{slug: entry} of the dataset registry (read only here), {} when absent."""
    p = Path(path) if path else registry_path()
    if not p.is_file():
        return {}
    reg = read_json(p, "dataset registry")
    ds = reg.get("datasets") if isinstance(reg.get("datasets"), dict) else reg
    return {str(k): v for k, v in (ds or {}).items() if isinstance(v, dict)}


def copy_scan_ready(cfg, inc=None):
    """(ready, evidence): True when one of the config's copy_scan.ready_files
    holds a passed calibration of the embedding copy detector (read through
    inc2.guard.load_calibration). No readable file, or no inc2 package: not
    ready (the P9 hold stays)."""
    from . import inc_dir
    files = ((cfg.raw.get("copy_scan") or {}).get("ready_files")) or []
    tried = []
    try:
        from ..inc2 import guard as G2
    except ImportError as e:
        return False, {"why": "inc2.guard is not importable (%s)" % e, "tried": files}
    for rel in files:
        p = inc_dir(inc) / rel
        if not p.is_file():
            tried.append({"path": str(p), "state": "missing"})
            continue
        try:
            rec = G2.load_calibration(p)
        except Exception as e:  # noqa: BLE001 - any refusal of the loader means "not passed"
            tried.append({"path": str(p), "state": "not passed", "why": str(e)[:200]})
            continue
        return True, {"path": str(p), "file": rec.get("file"), "cos_threshold": rec.get("cos_threshold")}
    return False, {"why": "no passed calibration", "tried": tried}


# ------------------------------------------------------------------ annotation
def annotation_of(cand, cfg):
    """(kind, evidence) of what a candidate's labels are: "boxes",
    "image_level" or "unknown". A provider's own statement wins; otherwise
    the text is read for box words and image-level words."""
    pf = cfg.raw["prefilter"]
    given = cand.get("annotation")
    ev = list(cand.get("annotation_evidence") or [])
    if given in ("boxes", "image_level"):
        return given, ev
    text = " ".join(str(cand.get(k) or "") for k in ("title", "description")) + " " + " ".join(
        str(x) for x in (cand.get("keywords") or []))
    t = " %s " % re.sub(r"[^a-z0-9]+", " ", text.lower())
    boxes = [w for w in pf["box_words"] if " %s" % re.sub(r"[^a-z0-9]+", " ", w.lower()).strip() in t]
    if boxes:
        return "boxes", ev + ["text:%s" % w for w in boxes]
    img = [w for w in pf["image_level_words"] if " %s" % re.sub(r"[^a-z0-9]+", " ", w.lower()).strip() in t]
    if img:
        return "image_level", ev + ["text:%s" % w for w in img]
    return given or "unknown", ev


# ------------------------------------------------------------------ decide
def _never_fetch(cand, cfg, never_train):
    pf = cfg.raw["prefilter"]
    sid = cand.get("source_id")
    if sid in never_train:
        return "the source is a never-train slug of the trainer"
    for r in pf.get("never_fetch") or []:
        if r.get("provider") == cand.get("provider") and str(r.get("ref", "")).lower() == str(cand.get("ref")).lower():
            return "the config lists %s:%s as never fetched (%s)" % (r["provider"], r["ref"], r.get("why", ""))
    title = " ".join(str(cand.get(k) or "") for k in ("title", "ref"))
    for rx in pf.get("never_fetch_title_regex") or []:
        if re.search(rx, title):
            return "the title matches a never-train dataset (%s)" % rx
    return None


def _estimate(cand, cfg, deficit, target_classes, class_rows):
    es = cfg.raw["estimates"]
    n_cls = len(class_rows) if class_rows else 0
    tb = None
    basis = None
    counted = [c for c in class_rows if c.get("cls") == "target" and isinstance(c.get("boxes"), (int, float))]
    if counted:
        tb = float(sum(c["boxes"] for c in counted))
        basis = "declared_box_counts"
    else:
        images = cand.get("images")
        share = (len([c for c in class_rows if c.get("cls") == "target"]) / float(n_cls)) if n_cls else \
            float(es["text_target_share"])
        if isinstance(images, (int, float)) and images > 0:
            tb = float(images) * float(es["boxes_per_image"]) * share
            basis = "images x boxes_per_image x target share"
        else:
            tb = float(es["unknown_images"]) * float(es["boxes_per_image"]) * share
            basis = "unknown size: estimates.unknown_images"
    b = cand.get("bytes")
    gb = (float(b) if isinstance(b, (int, float)) and b > 0 else float(es["unknown_bytes"])) / 1e9
    gb = max(gb, 0.01)
    dset = set(deficit or [])
    n_def = len([t for t in target_classes if t in dset])
    score = tb / gb * (1.0 + float(es["deficit_bonus"]) * n_def)
    return {"target_boxes": round(tb, 1), "basis": basis, "gb": round(gb, 4), "deficit_targets": n_def,
            "targets_per_gb": round(len(target_classes) / gb, 4), "score": round(score, 4)}


def known_role(known, source_id=None, provider=None, ref=None):
    """How a candidate relates to the known item it matched: "primary" (it is
    the item's own record: its source id, or its provider and ref), "copy"
    (a (provider, ref) the item lists under copies) or "match" (only one of
    the item's match rules, e.g. a title, matched: a re-upload or a
    derivative, which is not the item)."""
    if not known:
        return None
    if source_id is not None and source_id == known.get("id"):
        return "primary"
    if provider is not None and ref is not None and provider == known.get("provider") \
            and str(ref).lower() == str(known.get("ref")).lower():
        return "primary"
    for cp in known.get("copies") or []:
        if isinstance(cp, dict) and provider == cp.get("provider") and str(ref).lower() == str(cp.get("ref")).lower():
            return "copy"
    return "match"


def cleared_by_config(cfg, source_id, provider=None, ref=None):
    """True only when the source is the primary record of a known item that
    the config marks provenance_cleared, outside every evaluation lab (§3.2
    [review]: a primary release from a lab outside the evaluation labs). A
    re-upload that merely matches the item's title is never cleared, and
    nothing a candidate record carries can clear it."""
    it = cfg.known_item(source_id)
    if it is None or known_role(it, source_id, provider, ref) != "primary":
        for cand_it in cfg.known_items():
            if known_role(cand_it, source_id, provider, ref) == "primary":
                it = cand_it
                break
        else:
            return False
    if it.get("provenance_cleared") is not True:
        return False
    return not (it.get("lab_group") and it["lab_group"] in cfg.evaluation_labs())


# fields decide() derives; a stale copy of them (a candidates file from an earlier plan) never decides again
DERIVED = ("known_item", "known_item_id", "known_item_name", "known_item_role", "card", "class_status",
           "names_pending", "names_unresolved", "text_targets", "copy_candidate", "lab_group", "lab_group_basis",
           "evaluation_lab", "provenance_cleared", "hold_until", "exhaustive_labels", "licence", "decision",
           "target_classes", "estimate", "rank", "precheck", "superseded_by", "id", "found_by_search", "licence_ok",
           "licence_id", "licence_class", "expected_target_boxes", "credentials_ok", "image_level",
           "annotation_type", "copy_scan_done", "classes_source")


def decide(cand, cfg, names, targets, deficit=None, never_train=frozenset()):
    """The candidate with its class statuses, decision, lab group, holds,
    licence and estimate filled in (module docstring). Pure: no network.
    Provenance clearance is never read from the candidate: it comes only from
    the config, for the primary record of a cleared known item."""
    c = dict(cand)
    pf = cfg.raw["prefilter"]
    reasons = []
    known = cfg.known_item_for(c.get("source_id"), c.get("provider"), c.get("ref"), c.get("title"),
                               record_text(c))
    role = known_role(known, c.get("source_id"), c.get("provider"), c.get("ref"))
    c["known_item"] = bool(known)
    c["known_item_id"] = known["id"] if known else None
    c["known_item_name"] = known.get("name") if known else None
    c["known_item_role"] = role
    c.pop("provenance_cleared", None)
    if known and role == "primary" and "exhaustive_labels" in known:
        c["exhaustive_labels"] = known["exhaustive_labels"]
    card = cfg.card_table(c.get("source_id"))
    c["card"] = None if card is None else {"origin": card[2], "sha256": card[1]}
    classes = c.get("classes")
    if classes is None and known and role in ("primary", "copy") and known.get("classes"):
        classes = [dict(cl) for cl in known["classes"]]
        c["classes_source"] = "known_item"
    class_rows = [CM.map_class(cl, cfg, names, targets, card) for cl in (classes or [])]
    c["class_status"] = [{k: r.get(k) for k in ("src_id", "name", "status", "via", "taxon", "cls", "target",
                                                "basis", "reject_token", "pending", "boxes")}
                         for r in class_rows]
    pending = sorted({r["pending_name"] for r in class_rows if r["pending"]})
    tclasses = []
    for r in class_rows:
        if r["cls"] == "target" and r["target"] not in tclasses:
            tclasses.append(r["target"])
    text = " ".join(str(c.get(k) or "") for k in ("title", "description")) + " " + " ".join(
        str(x) for x in (c.get("keywords") or []))
    c["text_targets"] = targets.mentioned(text)
    if known and role in ("primary", "copy") and known.get("annotation") in ("boxes", "image_level"):
        c["annotation"] = known["annotation"]
        c["annotation_evidence"] = list(c.get("annotation_evidence") or []) + ["known_item:%s" % known["id"]]
    ann, ann_ev = annotation_of(c, cfg)
    c["annotation"], c["annotation_evidence"] = ann, ann_ev
    legacy = set(pf["legacy_labels"])
    legacy_equal = sum(1 for cl in (classes or []) if str(cl.get("name")) in legacy)
    c["copy_candidate"] = {"legacy_equal": legacy_equal} if legacy_equal >= int(pf["copy_candidate_min"]) else None
    # lab group: declared by the config, else presumed from the class list (§7.2 [review])
    lg, basis = cfg.lab_group_of(c.get("source_id"), c.get("provider"), c.get("ref"), c.get("title"),
                                 record_text(c))
    evaluation = set(cfg.evaluation_labs())
    # a known item's lab is the candidate's when the candidate is that item (or a listed copy); a record that
    # only matches its rules takes it only when it is an evaluation lab (the cautious reading of a copy)
    if known and known.get("lab_group") and (role in ("primary", "copy") or known["lab_group"] in evaluation):
        if not (lg in evaluation and known["lab_group"] not in evaluation):
            lg, basis = known["lab_group"], "declared"
    pd = pf.get("presumed_derivative") or {}
    declared = set(tclasses) | (set(c["text_targets"]) if not classes else set())
    if lg is None and c.get("provider") in (pd.get("providers") or []) and (
            declared & set(pd.get("declares_any") or []) or len(declared) >= int(pd.get("min_targets") or 10 ** 6)):
        lg, basis = pd.get("lab_group"), "presumed_derivative"
    c["lab_group"], c["lab_group_basis"] = lg, basis
    c["evaluation_lab"] = bool(lg and lg in evaluation)
    c["provenance_cleared"] = cleared_by_config(cfg, c.get("source_id"), c.get("provider"), c.get("ref")) \
        and not c["evaluation_lab"]
    c["hold_until"] = None if c["provenance_cleared"] else "h6_scan"
    # licence (P6)
    lic = c.get("licence")
    if not isinstance(lic, dict) or "class" not in lic:
        lt = c.get("licence_text")
        if lt is None and known and role in ("primary", "copy") and (known.get("licence") or {}).get("text"):
            lt = known["licence"]["text"]
        lic = LIC.gate(lt, cfg.raw["licence_policy"], evidence=c.get("licence_evidence"))
    c["licence"] = lic
    # the decision
    nf = _never_fetch(c, cfg, never_train)
    if nf:
        status, reasons = REJECTED, [{"code": "never_train", "detail": nf}]
    elif c["copy_candidate"]:
        status, reasons = COPY, [{"code": "copy_candidate", "detail": "%d class names equal the legacy labels"
                                  % legacy_equal}]
    elif tclasses:
        status, reasons = KEPT, [{"code": "target_class", "detail": ", ".join(tclasses)}]
    elif known and role in ("primary", "copy"):
        status, reasons = KEPT, [{"code": "known_item", "detail": known.get("name") or known["id"]}]
    elif card is not None:
        status, reasons = KEPT, [{"code": "card_table", "detail": card[2]}]
    elif classes and all(r["reject_token"] for r in class_rows):
        status, reasons = REJECTED, [{"code": "reject_token", "detail": sorted({r["reject_token"] for r in class_rows})}]
    elif pending:
        status, reasons = PENDING, [{"code": "names_pending", "detail": ", ".join(pending[:10])}]
    elif not classes and c["text_targets"] and ann == "boxes":
        status, reasons = KEPT, [{"code": "text_target", "detail": ", ".join(c["text_targets"])}]
        tclasses = list(c["text_targets"])
    elif not classes and c["text_targets"]:
        status, reasons = REJECTED, [{"code": "no_box_evidence", "detail": "the text names %s but shows no boxes"
                                      % ", ".join(c["text_targets"])}]
    else:
        status, reasons = REJECTED, [{"code": "no_target_class", "detail": "no declared class or text names a target"}]
    if status == KEPT and ann == "image_level":
        status = HELD
        reasons.append({"code": "image_level_only", "detail": "image-level labels only (filed R3 for a person)"})
    if status == KEPT and lic["class"] == "refused":
        status = REJECTED
        reasons.append({"code": "licence_refused", "detail": lic["id"]})
    c["target_classes"] = tclasses
    c["names_pending"] = pending
    c["decision"] = {"status": status, "reasons": reasons}
    c["estimate"] = _estimate(c, cfg, deficit, tclasses, class_rows)
    return c


def rank(cands):
    """Kept candidates by score (then targets per GB, then source id); the rest
    after them in source-id order. Sets "rank" (1..n) on the kept ones."""
    kept = [c for c in cands if c["decision"]["status"] == KEPT and not c.get("superseded_by")]
    rest = [c for c in cands if c not in kept]
    kept.sort(key=lambda c: (-c["estimate"]["score"], -c["estimate"]["targets_per_gb"], c["source_id"]))
    for i, c in enumerate(kept, 1):
        c["rank"] = i
    for c in rest:
        c["rank"] = None
    rest.sort(key=lambda c: c["source_id"])
    return kept + rest


def supersede(cands, cfg):
    """P6: among copies of one dataset, the preferred licence copy stays; the
    others get superseded_by and are closed. Copies are: a known item's own
    record and the copies it lists (a record that only matches one of its
    rules, e.g. by title, is not known to be a copy), or records with equal
    normalised titles AND the same declared image count (many unrelated
    projects share a generic title, a task name; a title alone would close a
    different dataset). Only kept candidates take part: a preferred copy
    the prefilter rejected would be fetched never, and would close the one
    that can be."""
    groups = {}
    for c in cands:
        if (c.get("decision") or {}).get("status") != KEPT:
            continue
        key = None
        ki = cfg.known_item(c.get("known_item_id")) if c.get("known_item_id") else None
        if ki is not None and known_role(ki, c.get("source_id"), c.get("provider"), c.get("ref")) not in (
                "primary", "copy"):
            ki = None
        if ki is None:
            for it in cfg.known_items():
                if known_role(it, c.get("source_id"), c.get("provider"), c.get("ref")) in ("primary", "copy"):
                    ki = it
        imgs = c.get("images")
        if ki is not None:
            key = "known:%s" % ki["id"]
        elif c.get("title") and isinstance(imgs, (int, float)) and not isinstance(imgs, bool) and imgs > 0:
            key = "title:%s:%d" % (re.sub(r"[^a-z0-9]+", "", str(c["title"]).lower()), int(imgs))
        if key:
            groups.setdefault(key, []).append(c)
    for key, grp in groups.items():
        if len(grp) < 2:
            continue
        best = grp[LIC.prefer([g["licence"] for g in grp])]
        for g in grp:
            if g is not best and g["licence"]["class"] != best["licence"]["class"]:
                g["superseded_by"] = best["source_id"]
                g["decision"]["reasons"].append({"code": "superseded", "detail": "a %s copy (%s) is preferred"
                                                 % (best["licence"]["class"], best["source_id"])})
    return cands


# ------------------------------------------------------------------ precheck
def _fail(out, code, detail, action, risk=None):
    out.append({"code": code, "detail": detail, "action": action, "risk": risk})


def precheck(c, cfg, ctx):
    """Every reason L16 may not fetch candidate c now (module docstring).
    ctx: {"state": {source: folded state}, "registry": {slug: entry},
    "never_train": set, "creds": {provider: (ok, detail)}, "copy_scan":
    (ready, evidence), "bytes_today": n, "bytes_total": n, "placement": doc or
    None, "max_bytes": n or None}."""
    f = []
    sid = c.get("source_id")
    prov = cfg.providers(enabled_only=True)
    if c.get("provider") not in prov:
        _fail(f, "provider_not_configured", "provider %r is not enabled in the config" % c.get("provider"), "refuse")
    nf = _never_fetch(c, cfg, ctx.get("never_train") or frozenset())
    if nf:
        _fail(f, "never_train", nf, "close")
    reg = (ctx.get("registry") or {}).get(sid)
    if isinstance(reg, dict):
        if str(reg.get("status")) == "quarantined":
            _fail(f, "quarantined", "the registry quarantines %s" % sid, "close")
        elif reg.get("annotation") != INTAKE_ANNOTATION:
            _fail(f, "registered_outside_intake", "%s is already a registry source (annotation %r): the stream "
                  "reads it through Step 1, the collector never registers it again" % (sid, reg.get("annotation")),
                  "close")
    st = (ctx.get("state") or {}).get(sid) or {}
    if st.get("status") in ("quarantined", "closed"):
        _fail(f, st["status"], "the source is %s (%s)" % (st["status"], st.get("reason")), "close")
    ki = cfg.known_item(c.get("known_item_id")) if c.get("known_item_id") else None
    if ki and ki.get("owned_by"):
        _fail(f, "owned_by_%s" % ki["owned_by"], ki.get("owned_why") or "another campaign owns this source", "hold")
    if ki and ki.get("fetchable") is False:
        _fail(f, "not_fetchable", ki.get("not_fetchable_why") or "the known item names no fetchable record",
              "hold", "R3")
    dec = (c.get("decision") or {}).get("status")
    if dec == COPY:
        _fail(f, "copy_candidate", "a probable re-export of the reference dataset", "close")
    elif dec == PENDING:
        _fail(f, "names_pending", "class names await lever L26 (collect names)", "hold")
    elif dec == REJECTED:
        codes = ",".join(r["code"] for r in (c.get("decision") or {}).get("reasons") or [])
        _fail(f, "no_target_class" if "no_target" in codes or "reject" in codes or "box" in codes else codes,
              "the prefilter rejected it (%s)" % codes, "hold", "R3")
    if c.get("annotation") == "image_level":
        _fail(f, "image_level_only", "image-level labels only", "hold", "R3")
    if c.get("superseded_by"):
        _fail(f, "superseded", "a preferred licence copy exists (%s)" % c["superseded_by"], "close")
    lic = c.get("licence") or {}
    if lic.get("class") == "unresolved":
        _fail(f, "licence_unresolved", "licence %r is unresolved (P6; card X16)" % lic.get("id"), "hold", "R3")
    elif lic.get("class") == "refused":
        _fail(f, "licence_refused", "licence %r is not research-usable" % lic.get("id"), "close")
    if c.get("evaluation_lab") and c.get("lab_group_basis") == "declared":
        ready, ev = ctx.get("copy_scan") or (False, None)
        if not ready:
            _fail(f, "copy_scan_pending", "lab group %s is an evaluation lab; held until the copy detector is "
                  "calibrated (P9)" % c.get("lab_group"), "hold", "R3")
    ok, detail = (ctx.get("creds") or {}).get(c.get("provider"), (True, None))
    if not ok:
        _fail(f, "credentials_missing", detail or "provider credentials are missing (card X16)", "hold", "R3")
    bu = cfg.budgets()
    est = c.get("bytes")
    cap = float((bu.get("approved_bytes") or {}).get(sid) or bu["bytes_per_source"])
    done = float(st.get("bytes") or 0)
    if isinstance(est, (int, float)) and est > 0 and done + est > cap:
        _fail(f, "over_source_cap", "%.2f GB fetched + %.2f GB planned > the %.0f GB per-source cap (a person may "
              "approve more)" % (done / 1e9, est / 1e9, cap / 1e9), "hold", "R3")
    if float(ctx.get("bytes_today") or 0) >= float(bu["bytes_daily"]):
        _fail(f, "daily_bytes", "the daily byte cap (%.0f GB) is reached" % (bu["bytes_daily"] / 1e9), "hold")
    if float(ctx.get("bytes_total") or 0) >= float(bu["bytes_envelope"]):
        _fail(f, "byte_envelope", "the byte envelope (%.0f GB) is spent" % (bu["bytes_envelope"] / 1e9), "hold", "R3")
    if int(st.get("failed_attempts") or 0) >= int(bu["attempts_per_source"]):
        _fail(f, "attempts_exhausted", "%d failed attempts" % st["failed_attempts"], "close")
    if in_slurm() and not placed_on_cluster(c.get("provider"), ctx.get("placement"), cfg):
        _fail(f, "not_placed_on_cluster", "provider %s is not placed on compute nodes (placement.json); the lab "
              "hook fetches it" % c.get("provider"), "refuse")
    risk = "R3" if any(x.get("risk") == "R3" for x in f) else None
    return {"ok": not f, "risk": risk, "failures": f,
            "action": max((x["action"] for x in f), key=lambda a: ACTION_RANK[a]) if f else None}


def placed_on_cluster(provider, placement, cfg):
    """True when the network probe (placement.json) passed for the provider
    from a compute node and the config does not keep it on the lab."""
    try:
        cfg.provider(provider)
    except ConfigError:
        return False
    if cfg.lab_only(provider):
        return False
    p = ((placement or {}).get("providers") or {}).get(provider) or {}
    return bool((placement or {}).get("in_slurm")) and p.get("placement") == "cluster"


def refusal_from(check):
    """A Refusal carrying every failure of a precheck result."""
    fs = check["failures"]
    worst = max(fs, key=lambda x: ACTION_RANK[x["action"]])
    return Refusal(worst["code"], "; ".join("%s: %s" % (x["code"], x["detail"]) for x in fs),
                   action=worst["action"], risk=check["risk"], failures=fs)
