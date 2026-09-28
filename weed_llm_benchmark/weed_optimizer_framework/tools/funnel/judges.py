"""The machine judges of the funnel audit: J-zs, J-knn1, J-knn2, and the
Step 1 probe (J1) on the independent-truth photos (contract
docs/FUNNEL_AUDIT.md §4.3, DEC-9; runner docs/FUNNEL_AUDIT_RUNNER.md §4.12,
§5.3.2, §7.2).

  J-zs    zero-shot over a closed candidate set with the Step 1 embedder's
          own text tower: one prompt per target (its lineage from the
          taxonomy resolver and its common name, in the config's template),
          one per attractor and per named non-target (both collapse to
          "other"), and the config's non-object prompts. P is the softmax of
          scale * cosine over every prompt, summed per label. The features
          are the Step 1 crop features (the adapter's), never re-embedded.
          The text tower is the adapter's (injected); this file never names
          its class.
  J-knn1  a kNN over the reference crops (bank KT1) with DINOv2 features;
          a query never sees a bank crop of its own capture session, nor of
          its provenance group (so a copy of a reference photograph never
          sees its twin). Label space: the targets.
  J-knn2  a kNN over a multi-domain bank (KT1, KT4, KT5 capped per
          (source, name) cell, KT6 calibration half, KT7 exemplars). For
          every query, every bank entry that shares a source, a 3-bit
          near-duplicate group, a provenance group or an annotating lab with
          it is excluded: the disjointness rule of contract §4.1. The keys
          are read as qualify reads them (qualify.item_keys: an item of an
          independent set is compared per observation) and compared by the
          rule of qualify.shares, vectorised (any kind equal, a missing value
          never equal). Label space: the targets and "other".
  J1      the Step 1 probe on the independent-truth photos
          (judges/J1__kt7.npz), through the adapter, on features of those
          photos made by the adapter's Step 1 embedder.

kNN rule: the top k eligible neighbours by cosine, weights exp(cos / tau),
normalised; P is the weighted vote. A query with no eligible neighbour, or
without a feature, is a NaN row (top -1).

Score files judges/<judge>__<set>.npz (set = crops, the Step 1 crop table,
or kt7): unit_index int64 (crop_id), P float16 [n, L], top int16, and meta, a
JSON string with the runner §1.2 header plus judge, labels, set, features,
bank, exclusion, k, temperature, prompts and crops_sha256. A rerun on the
same inputs, parameters and code is a no-op; any difference refuses unless
force.

Nothing here names a domain.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from . import (FUNNEL_DIR, JudgeError, canonical_json, header, sha256_bytes, strip_volatile,
               write_json_atomic)
from . import embed as E
from . import qualify as QF
from ..inc import common as C
from ..inc import verify as V

KINDS = ("step1_probe", "zero_shot", "knn", "rl")
DISJOINT_KEYS = tuple(QF.KEYS)                  # the four kinds of qualify.shares (runner §1.3, §7.2)
KEY_KINDS = DISJOINT_KEYS + ("session",)
EXCLUSIONS = ("disjoint", "session", "none")
OTHER, NON_OBJECT = "other", "non_object"
SETS = ("crops", "kt7")
# unit-id prefix of each crop-table set (runner §1.3)
UNIT_PREFIX = {"core": "t1", "copy": "t2", "pool": "b"}
CHUNK = 2048
ZS_CHUNK = 65536
SCORE_FORMAT = "funnel-judge-scores/1"
INDEX_FORMAT = "funnel-judge-index/1"
KT7_TABLE = Path("kt7") / "crops_kt7.csv"
DINO_DIR = "emb_dinov2"
DINO_KT7 = "emb_dinov2_kt7.npz"


def log(msg):
    print("[funnel.judges] %s" % msg, flush=True)


def _raw(domain):
    raw = getattr(domain, "raw", domain)
    if not isinstance(raw, dict):
        raise JudgeError("a domain config (or its raw dict) is required, got %r" % (domain,))
    return raw


def _targets(domain):
    t = list(_raw(domain)["classes"]["targets"])
    if [x["id"] for x in t] != list(range(len(t))):
        raise JudgeError("the target ids are not 0..n-1 in config order")
    return t


def panel(domain):
    """{judge id: panel entry} of the config's judges.panel, kinds checked."""
    entries = (_raw(domain).get("judges") or {}).get("panel")
    if not isinstance(entries, list) or not entries:
        raise JudgeError("the domain config has no judges.panel")
    out = {}
    for e in entries:
        if not isinstance(e, dict) or not e.get("id") or e.get("kind") not in KINDS:
            raise JudgeError("judges.panel entry %r: an id and a kind in %s are required" % (e, KINDS))
        if e["id"] in out:
            raise JudgeError("judges.panel: duplicate id %r" % e["id"])
        out[e["id"]] = dict(e)
    return out


# ------------------------------------------------------------ zero-shot
def zero_shot(X, T, scale, groups, n_labels):
    """P [n, n_labels]: the softmax of scale * cosine(X, T) over every prompt,
    summed per label (groups[i] is prompt i's label). A row of X that is not
    finite gives a NaN row."""
    X = np.asarray(X, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    if T.ndim != 2 or X.ndim != 2 or T.shape[1] != X.shape[1]:
        raise JudgeError("zero_shot: features %s and prompts %s disagree" % (X.shape, T.shape))
    if len(groups) != len(T):
        raise JudgeError("zero_shot: %d prompts, %d label assignments" % (len(T), len(groups)))
    if len(groups) == 0 or groups.min() < 0 or groups.max() >= int(n_labels):
        raise JudgeError("zero_shot: a prompt label outside 0..%d" % (int(n_labels) - 1))
    missing = sorted(set(range(int(n_labels))) - set(groups.tolist()))
    if missing:
        raise JudgeError("zero_shot: label(s) %s have no prompt" % missing)
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0:
        raise JudgeError("zero_shot: the logit scale %r is not a positive number" % scale)
    tn = np.linalg.norm(T, axis=1, keepdims=True)
    if not np.isfinite(T).all() or (tn <= 0).any():
        raise JudgeError("zero_shot: a prompt feature is not finite or is zero")
    T = T / tn
    M = np.zeros((len(groups), int(n_labels)), dtype=np.float64)
    M[np.arange(len(groups)), groups] = 1.0
    n = len(X)
    P = np.full((n, int(n_labels)), np.nan, dtype=np.float64)
    ok = np.isfinite(X).all(axis=1)
    idx = np.flatnonzero(ok)
    for s in range(0, len(idx), ZS_CHUNK):
        rows = idx[s:s + ZS_CHUNK]
        Xc = X[rows]
        Xc = Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-12)
        L = scale * (Xc @ T.T)
        L -= L.max(axis=1, keepdims=True)
        Ex = np.exp(L)
        Ex /= Ex.sum(axis=1, keepdims=True)
        P[rows] = Ex @ M
    return P


def prompts(domain, resolver):
    """(texts, label_of_text, labels): one prompt per target in config order
    (label = its id), per attractor and per named non-target that is not an
    attractor (label "other"), and the config's non-object prompts (label
    "non_object"). The template is judges.panel[zero_shot].prompt_template,
    formatted with the resolver's lineage string and the common name (a named
    non-target has no common name in the config: its taxon stands in)."""
    raw = _raw(domain)
    zs = [e for e in panel(domain).values() if e["kind"] == "zero_shot"]
    if len(zs) != 1:
        raise JudgeError("judges.panel must hold exactly one zero_shot judge, holds %d" % len(zs))
    zs = zs[0]
    template = zs.get("prompt_template")
    if not isinstance(template, str) or "{lineage}" not in template:
        raise JudgeError("the zero_shot judge has no prompt_template with {lineage}")
    collapse = zs.get("collapse")
    if not isinstance(collapse, dict) or set(collapse) != {"attractors", "named_non_targets"} \
            or any(v != OTHER for v in collapse.values()):
        raise JudgeError("zero_shot collapse must map attractors and named_non_targets to %r, got %r"
                         % (OTHER, collapse))
    non_object = zs.get("non_object_prompts")
    if not isinstance(non_object, list) or not non_object or not all(isinstance(p, str) and p for p in non_object):
        raise JudgeError("the zero_shot judge has no non_object_prompts")
    targets = _targets(domain)
    n_t = len(targets)
    labels = [t["name"] for t in targets] + [OTHER, NON_OBJECT]

    def fmt(taxon, common):
        try:
            lineage = resolver.lineage_string(taxon)
        except Exception as e:  # noqa: BLE001 - the resolver's own refusal, named
            raise JudgeError("no lineage for %r (%s: %s); run fetch --what taxonomy" % (taxon, type(e).__name__, e))
        if not lineage:
            raise JudgeError("an empty lineage for %r" % taxon)
        return template.format(lineage=lineage, common=common)

    texts, groups = [], []
    seen = set()
    for t in targets:
        if not t.get("taxon") or not t.get("common"):
            raise JudgeError("target %s needs a taxon and a common name" % t.get("name"))
        texts.append(fmt(t["taxon"], t["common"]))
        groups.append(int(t["id"]))
        seen.add(t["taxon"])
    for a in raw.get("attractors", []) or []:
        if a["taxon"] in seen:
            continue
        seen.add(a["taxon"])
        texts.append(fmt(a["taxon"], a.get("common") or a["taxon"]))
        groups.append(n_t)
    for t in targets:
        for taxon in t.get("not", []) or []:
            if taxon in seen:
                continue
            seen.add(taxon)
            texts.append(fmt(taxon, taxon))
            groups.append(n_t)
    for p in non_object:
        texts.append(p)
        groups.append(n_t + 1)
    if len(set(texts)) != len(texts):
        raise JudgeError("two zero-shot prompts are the same text")
    return texts, groups, labels


def text_features(encoder, texts, dim):
    T = np.asarray(encoder(list(texts)), dtype=np.float64)
    if T.shape != (len(texts), int(dim)):
        raise JudgeError("the text tower returned %s for %d prompts; the crop features have dim %d"
                         % (T.shape, len(texts), int(dim)))
    if not np.isfinite(T).all() or (np.linalg.norm(T, axis=1) <= 0).any():
        raise JudgeError("the text tower returned a non-finite or zero feature")
    return T


# ------------------------------------------------------------------ kNN
def _columns(keys, n, what):
    """{kind: list of n values} from a column dict or a list of per-row dicts."""
    if isinstance(keys, dict):
        out = {}
        for kind in KEY_KINDS:
            vals = keys.get(kind)
            if vals is None:
                out[kind] = [None] * n
            else:
                vals = list(vals)
                if len(vals) != n:
                    raise JudgeError("%s keys: %s has %d values for %d rows" % (what, kind, len(vals), n))
                out[kind] = vals
        return out
    rows = list(keys)
    if len(rows) != n:
        raise JudgeError("%s keys: %d rows for %d features" % (what, len(rows), n))
    return {kind: [r.get(kind) for r in rows] for kind in KEY_KINDS}


def _codes(q_vals, b_vals):
    """Integer codes over one vocabulary; a missing value never equals anything."""
    vocab = {}

    def enc(vals, missing):
        out = np.empty(len(vals), dtype=np.int64)
        for i, v in enumerate(vals):
            if v is None or v == "":
                out[i] = missing
            else:
                out[i] = vocab.setdefault(str(v), len(vocab))
        return out
    return enc(q_vals, -1), enc(b_vals, -2)


def knn(Q, q_keys, bank_X, bank_labels, bank_keys, n_labels, k=10, temperature=0.07, exclude="disjoint",
        chunk=CHUNK):
    """P float32 [n, n_labels]: for each query, the top k bank entries by
    cosine among those it may see, weighted exp(cos / temperature), summed
    per label and normalised.

    exclude="disjoint": a bank entry that shares a source, near_dup3,
    provenance or lab value with the query is never seen (contract §4.1).
    exclude="session": an entry of the query's own capture session (when the
    query has one) or of its provenance group is never seen. "none": all are
    seen. Keys are {kind: values} columns or per-row dicts."""
    if exclude not in EXCLUSIONS:
        raise JudgeError("exclude %r is not one of %s" % (exclude, EXCLUSIONS))
    k = int(k)
    tau = float(temperature)
    if k < 1 or not np.isfinite(tau) or tau <= 0:
        raise JudgeError("knn needs k >= 1 and a positive temperature (k=%r, temperature=%r)" % (k, temperature))
    Q = np.asarray(Q, dtype=np.float32)
    B = np.asarray(bank_X, dtype=np.float32)
    labels = np.asarray(bank_labels, dtype=np.int64)
    n, m = len(Q), len(B)
    if m == 0:
        raise JudgeError("knn: an empty bank")
    if B.ndim != 2 or Q.ndim != 2 or Q.shape[1] != B.shape[1] or len(labels) != m:
        raise JudgeError("knn: queries %s, bank %s, labels %d disagree" % (Q.shape, B.shape, len(labels)))
    if not np.isfinite(B).all():
        raise JudgeError("knn: a bank feature is not finite")
    if labels.min() < 0 or labels.max() >= int(n_labels):
        raise JudgeError("knn: a bank label outside 0..%d" % (int(n_labels) - 1))
    qk = _columns(q_keys, n, "query")
    bk = _columns(bank_keys, m, "bank")
    codes = {kind: _codes(qk[kind], bk[kind]) for kind in KEY_KINDS}
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    P = np.full((n, int(n_labels)), np.nan, dtype=np.float32)
    ok = np.isfinite(Q).all(axis=1)
    idx_ok = np.flatnonzero(ok)
    kk = min(k, m)
    for s in range(0, len(idx_ok), int(chunk)):
        rows = idx_ok[s:s + int(chunk)]
        Qc = Q[rows]
        Qc = Qc / np.maximum(np.linalg.norm(Qc, axis=1, keepdims=True), 1e-12)
        S = Qc @ Bn.T
        excluded = np.zeros(S.shape, dtype=bool)
        if exclude == "disjoint":  # funnel-mutation: M7
            for kind in DISJOINT_KEYS:
                cq, cb = codes[kind]
                excluded |= cb[None, :] == cq[rows][:, None]
        if exclude == "session":
            cq, cb = codes["session"]
            excluded |= cb[None, :] == cq[rows][:, None]
            cq, cb = codes["provenance"]
            excluded |= cb[None, :] == cq[rows][:, None]
        S[excluded] = -np.inf
        if kk < m:
            top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        else:
            top = np.tile(np.arange(m), (len(rows), 1))
        vals = np.take_along_axis(S, top, axis=1)
        valid = np.isfinite(vals)
        vmax = np.where(valid, vals, -np.inf).max(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", over="ignore"):
            w = np.where(valid, np.exp((vals - np.where(np.isfinite(vmax), vmax, 0.0)) / tau), 0.0)
        tot = w.sum(axis=1)
        Pc = np.zeros((len(rows), int(n_labels)), dtype=np.float64)
        r_idx = np.repeat(np.arange(len(rows)), top.shape[1])
        np.add.at(Pc, (r_idx, labels[top].ravel()), w.ravel())
        with np.errstate(invalid="ignore", divide="ignore"):
            Pc = Pc / tot[:, None]
        Pc[tot <= 0] = np.nan
        P[rows] = Pc.astype(np.float32)
    return P


# ------------------------------------------------------------------ banks
def _crop_sets(crop_table):
    s = crop_table.set
    if isinstance(s, np.ndarray) and s.dtype.kind in "iu":
        return [V.SETS[int(i)] for i in s]
    return [str(x) for x in s]


def crop_unit_ids(crop_table):
    """The runner §1.3 unit id of every crop-table row (t1: reference crop,
    t2: copy crop, b: pool box)."""
    sets = _crop_sets(crop_table)
    out = []
    for i, st in enumerate(sets):
        pre = UNIT_PREFIX.get(st)
        if pre is None:
            raise JudgeError("crop %d: set %r has no unit-id prefix (%s)" % (i, st, sorted(UNIT_PREFIX)))
        out.append("%s:%s#%d" % (pre, crop_table.key[i], int(crop_table.box[i])))
    return out


def item_keys(item, domain):
    """The disjointness keys of a known-truth item as qualify reads them (an
    item of an independent set is compared per observation), plus its
    capture session."""
    if hasattr(domain, "independent_sets"):
        keys = QF.item_keys(item, domain)
    else:
        kts = item.get("kt")
        kts = [kts] if isinstance(kts, str) else list(kts or [])
        kt_cfg = _raw(domain).get("known_truth") or {}
        ind = any((kt_cfg.get(k) or {}).get("independent") is True for k in kts if k)
        keys = QF.item_keys(item, None, independent=ind)
    keys["session"] = item.get("session")
    return keys


def _bank_specs(entry):
    specs = entry.get("bank")
    if not isinstance(specs, list) or not specs:
        raise JudgeError("knn judge %s has no bank list" % entry.get("id"))
    out = []
    for sp in specs:
        kt, _, role = str(sp).partition(":")
        out.append((kt, role or None))
    return out


_CAP_RE = re.compile(r"^(kt\d+)_per_cell_max$")


def _cell_caps(entry, specs, jid):
    """{KT id: cap} from the entry's "<kt>_per_cell_max" keys; the set must be
    in the bank."""
    caps = {}
    in_bank = {kt for kt, _r in specs}
    for key, val in entry.items():
        m = _CAP_RE.match(str(key))
        if not m:
            continue
        kt = m.group(1).upper()
        if kt not in in_bank:
            raise JudgeError("knn judge %s: %s caps %s, which is not in its bank" % (jid, key, kt))
        if not isinstance(val, int) or isinstance(val, bool) or val < 1:
            raise JudgeError("knn judge %s: %s must be a positive integer" % (jid, key))
        caps[kt] = int(val)
    return caps


def _seed_tag(judge_id):
    """The seed-text tag of a judge id: lower case, without a leading "j-"."""
    s = str(judge_id).lower()
    return s[2:] if s.startswith("j-") else s


def _item_label(item, n_targets, with_other, judge_id):
    kind = item.get("truth_kind")
    if kind == "target":
        t = item.get("truth")
        if t is None or not 0 <= int(t) < n_targets:
            raise JudgeError("%s bank: target item %s has truth %r" % (judge_id, item.get("id"), t))
        return int(t)
    if kind in ("attractor", "other"):
        return n_targets if with_other else None
    return None


def banks(domain, adapter, features, name_status=None, known_truth=None, crop_table=None, funnel_dir=None):
    """{knn judge id: {"X" float32, "labels", "keys" (columns), "kt", "ids",
    "n_labels", "label_names", "exclusion", "k", "temperature", "sha256",
    "dropped"}}.

    features = {"crops": X over the crop table, "kt7": X over the KT7 table
    (either may be None when no bank entry needs it)}. A "<KT>:<role>" spec
    keeps the items of that role. An entry key "<kt>_per_cell_max" caps that
    set at so many items per (source, source class name) cell, drawn with
    seed funnel/v1/<judge tag>/<kt>/<cell> (for J-knn2 and its KT5 cap:
    funnel/v1/knn2/kt5/<cell>, runner §5.3.2). An item without a finite
    feature is dropped and counted. name_status (the name_status_v2 record
    that fixes KT membership upstream) is recorded in the bank digest when
    given."""
    targets = _targets(domain)
    n_t = len(targets)
    known = known_truth if known_truth is not None else adapter.known_truth(domain, funnel_dir or FUNNEL_DIR)
    table = crop_table
    out = {}
    for jid, entry in sorted(panel(domain).items()):
        if entry["kind"] != "knn":
            continue
        excl = entry.get("holdout")
        exclusion = {"session": "session", "disjoint": "disjoint"}.get(excl)
        if exclusion is None:
            raise JudgeError("knn judge %s: holdout %r is not session or disjoint" % (jid, excl))
        specs = _bank_specs(entry)
        items, empty = [], []
        for kt, role in specs:
            if kt not in known:
                raise JudgeError("knn judge %s: the adapter has no known-truth set %s" % (jid, kt))
            sel = [it for it in known[kt] if role is None or it.get("role") == role]
            if not sel:
                # membership can be conditional (a set of claimed labels enters only after the
                # hypothesis that tests them passes); the empty spec is recorded, not guessed around
                empty.append(kt + (":" + role if role else ""))
                continue
            items.extend((kt, it) for it in sorted(sel, key=lambda it: it["id"]))
        with_other = entry.get("labels")
        if with_other is None:
            with_other = any(it.get("truth_kind") in ("attractor", "other") for _kt, it in items)
        else:
            if with_other not in ("targets", "targets+other"):
                raise JudgeError("knn judge %s: labels %r" % (jid, with_other))
            with_other = with_other == "targets+other"
        seeds = {}
        caps = _cell_caps(entry, specs, jid)
        if caps and table is None:
            table = adapter.crop_table()
        for cap_kt, cap in sorted(caps.items()):
            cells = {}
            for kt, it in items:
                if kt != cap_kt:
                    continue
                cid = it.get("crop_id")
                if cid is None or it.get("crop_set") == "kt7":
                    raise JudgeError("knn judge %s: %s item %s has no crop-table row" % (jid, kt, it["id"]))
                cell = "%s|%s" % (it.get("source"), table.src_name[int(cid)])
                cells.setdefault(cell, []).append(it["id"])
            keep = set()
            for cell, ids in sorted(cells.items()):
                ids = sorted(ids)
                if len(ids) > cap:
                    text = "funnel/v1/%s/%s/%s" % (_seed_tag(jid), cap_kt.lower(), cell)
                    seeds["%s/%s" % (cap_kt, cell)] = text
                    perm = np.random.default_rng(C.stable_int(text)).permutation(len(ids))
                    ids = sorted(ids[i] for i in perm[:cap])
                keep.update(ids)
            items = [(kt, it) for kt, it in items if kt != cap_kt or it["id"] in keep]
        seen, rows = {}, []
        dropped = {"no_label": 0, "no_feature": 0, "duplicate": 0}
        for kt, it in items:
            lab = _item_label(it, n_t, with_other, jid)
            if lab is None:
                dropped["no_label"] += 1
                continue
            if it["id"] in seen:
                if seen[it["id"]] != lab:
                    raise JudgeError("knn judge %s: unit %s in two sets with labels %s and %s"
                                     % (jid, it["id"], seen[it["id"]], lab))
                dropped["duplicate"] += 1
                continue
            cs = "kt7" if it.get("crop_set") == "kt7" else "crops"
            X = features.get(cs)
            cid = it.get("crop_id")
            if X is None or cid is None:
                raise JudgeError("knn judge %s: no %s feature for %s" % (jid, cs, it["id"]))
            x = np.asarray(X[int(cid)], dtype=np.float32)
            if not np.isfinite(x).all():
                dropped["no_feature"] += 1
                continue
            seen[it["id"]] = lab
            rows.append((kt, it, lab, x))
        if not rows:
            raise JudgeError("knn judge %s: no bank entry has a label and a feature" % jid)
        kd = [item_keys(it, domain) for _kt, it, _l, _x in rows]
        keys = {kind: [d.get(kind) for d in kd] for kind in KEY_KINDS}
        if exclusion == "session" and any(v in (None, "") for v in keys["session"]):
            raise JudgeError("knn judge %s: holdout by session needs a session on every bank entry" % jid)
        for kind in DISJOINT_KEYS:
            if exclusion == "disjoint" and any(v in (None, "") for v in keys[kind]):
                raise JudgeError("knn judge %s: a bank entry has no %s key" % (jid, kind))
        ids = [it["id"] for _kt, it, _l, _x in rows]
        labs = [lab for _kt, _it, lab, _x in rows]
        body = {"ids": ids, "labels": labs, "kt": [kt for kt, _it, _l, _x in rows],
                "keys": keys, "name_status": (name_status or {}).get("sha256"), "empty": empty}
        out[jid] = {"X": np.stack([x for _kt, _it, _l, x in rows]), "labels": np.array(labs, dtype=np.int64),
                    "ids": ids, "keys": keys, "kt": sorted({kt for kt, _it, _l, _x in rows}), "specs": entry["bank"],
                    "n_labels": n_t + (1 if with_other else 0),
                    "label_names": [t["name"] for t in targets] + ([OTHER] if with_other else []),
                    "exclusion": exclusion, "k": int(entry.get("k", 10)),
                    "temperature": float(entry.get("temperature", 0.07)),
                    "sha256": sha256_bytes(canonical_json(body).encode("utf-8")),
                    "dropped": dropped, "seeds": seeds, "n": len(rows), "empty": empty}
    return out


# ------------------------------------------------------------ score files
def _score_path(funnel_dir, judge, set_name):
    return Path(funnel_dir) / "judges" / ("%s__%s.npz" % (judge, set_name))


def _read_meta(path):
    try:
        with np.load(path) as d:
            return json.loads(str(d["meta"]))
    except Exception:  # noqa: BLE001 - an unreadable file is a file to rebuild or refuse on
        return None


def _identity(meta):
    """A score file's identity: its meta without volatile keys, with the prereg
    compared by its core (runner §3.3: an amendment such as the sample lock
    leaves earlier artifacts current), and the contract and the config by
    their sha256."""
    out = {k: v for k, v in meta.items() if k not in ("stats", "prereg", "contract", "domain_config")}
    out["prereg_core_sha256"] = (meta.get("prereg") or {}).get("core_sha256")
    out["contract_sha256"] = (meta.get("contract") or {}).get("sha256")
    out["domain_config_sha256"] = (meta.get("domain_config") or {}).get("sha256")
    return strip_volatile(out)


def keys_sha256(columns, block=50000):
    """sha256 of query key columns {kind: values} in canonical JSON, kind by
    kind in sorted order, block by block."""
    import hashlib
    h = hashlib.sha256()
    for kind in sorted(columns):
        vals = list(columns[kind])
        h.update(("%s|%d|" % (kind, len(vals))).encode("utf-8"))
        for s in range(0, len(vals), int(block)):
            h.update(canonical_json([None if v is None else str(v) for v in vals[s:s + int(block)]]).encode("utf-8"))
    return h.hexdigest()


def array_sha256(X, block_rows=65536):
    """sha256 over an array's dtype, shape and values, block by block (no
    whole copy of a large feature matrix)."""
    import hashlib
    X = np.asarray(X)
    h = hashlib.sha256(("%s|%s|" % (X.dtype.str, list(X.shape))).encode("utf-8"))
    for s in range(0, len(X), int(block_rows)):
        h.update(np.ascontiguousarray(X[s:s + int(block_rows)]).tobytes())
    return h.hexdigest()


def _write_scores(path, meta, unit_index, P, extra_arrays=None, force=False):
    """Write one score file; a current file (same identity) is a no-op and a
    different one refuses unless force. Returns {"path", "sha256", "n", "noop"}."""
    path = Path(path)
    if path.exists():
        old = _read_meta(path)
        if old is not None and _identity(old) == _identity(meta):
            return {"path": str(path), "sha256": C.sha256_file(path), "n": int(len(unit_index)), "noop": True}
        if not force:
            raise JudgeError("%s exists and was made from other inputs, parameters or code; rerun with --force"
                             % path)
    P = np.asarray(P, dtype=np.float32)
    top = np.where(np.isfinite(P).all(axis=1), np.nan_to_num(P, nan=-1.0).argmax(axis=1), -1).astype(np.int16)
    arrays = {"unit_index": np.asarray(unit_index, dtype=np.int64), "P": P.astype(np.float16), "top": top}
    arrays.update(extra_arrays or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    V._save_npz(path, meta, **arrays)
    return {"path": str(path), "sha256": C.sha256_file(path), "n": int(len(unit_index)), "noop": False}


def _would_noop(path, meta):
    path = Path(path)
    if not path.exists():
        return False
    old = _read_meta(path)
    return old is not None and _identity(old) == _identity(meta)


def _adapter_embedder(adapter):
    """(tag, factory) of the adapter's Step 1 image embedder: the one adapter
    interface function whose name ends in "_embedder"."""
    try:
        from .adapters import INTERFACE
        names = [n for n in INTERFACE if n.endswith("_embedder")]
    except ImportError:
        names = [n for n in dir(adapter) if n.endswith("_embedder") and not n.startswith("_")]
    names = [n for n in names if callable(getattr(adapter, n, None))]
    if len(names) != 1:
        raise JudgeError("the adapter must offer exactly one *_embedder function, found %s" % names)
    return names[0][:-len("_embedder")], getattr(adapter, names[0])


def _kt7_features(funnel_dir, table, out_name, want_name, factory):
    """(X, meta) of the KT7 table's features by the embedder named want_name,
    embedding them when the file is missing or stale."""
    path = Path(funnel_dir) / out_name
    meta = E._npz_meta(path) if path.exists() else None
    if not (meta and meta.get("crops_sha256") == table.sha and meta.get("embedder") == want_name):
        emb = factory()
        if emb.name != want_name:
            raise JudgeError("the embedder for %s is %s, expected %s" % (out_name, emb.name, want_name))
        E.embed_table(table.path, path, emb, force=meta is not None)
    return E.load_table(table, path, embedder=want_name)


def _unit_keys(adapter, unit_ids, domain, funnel_dir):
    """adapter.unit_keys(unit_ids), passing the domain and the funnel directory
    when the adapter takes them, so the keys and the known truth are read
    from the same funnel directory."""
    import inspect
    try:
        params = inspect.signature(adapter.unit_keys).parameters
    except (TypeError, ValueError):
        params = {}
    kw = {k: v for k, v in (("domain", domain), ("funnel_dir", funnel_dir)) if k in params}
    return adapter.unit_keys(unit_ids, **kw)


def _feature_record(path, sha):
    return {"path": str(path), "sha256": sha}


def _shard_set_record(info):
    """A feature shard set (embed.load's info): a directory is no file a
    consumer can re-hash, so the record has no path; it names the directory
    and every shard file's sha256, and its sha256 is their digest."""
    return {"path": None, "dir": str(info["path"]), "files": dict(sorted(info["files"].items())),
            "sha256": info["sha256"]}


def score_all(prereg, domain, funnel_dir, adapter, text_encoder=None, resolver=None, dinov2_embedder=None,
              step1_embedder=None, force=False, testing=False):
    """Every judge score file of the panel over the crop table and the KT7
    table, plus judges/index.json. Returns the index."""
    funnel_dir = Path(funnel_dir)
    t0 = time.time()
    pnl = panel(domain)
    targets = _targets(domain)
    n_t = len(targets)
    crops = adapter.crop_table()
    known = adapter.known_truth(domain, funnel_dir)
    model, pooling = E.features_config(domain)
    dino_name = E.embedder_name(model, pooling)
    X_dino, dino_info = E.load(crops, funnel_dir / DINO_DIR, embedder=dino_name)
    kt7_path = funnel_dir / KT7_TABLE
    if not kt7_path.is_file():
        raise JudgeError("%s not found: the independent-truth photos are fetched first (fetch --what kt7, "
                         "lever L11a)" % kt7_path)
    kt7 = E.CropTable(kt7_path)
    kt7_items = sorted(known.get("KT7") or [], key=lambda it: it["id"])
    if not kt7_items:
        raise JudgeError("the adapter returned no KT7 items; the judges are qualified on them (contract §4.1)")
    no_row = [it["id"] for it in kt7_items if it.get("crop_id") in (None, "")]
    if no_row:
        raise JudgeError("%d KT7 item(s) have no row in %s (e.g. %s); rerun fetch --what kt7"
                         % (len(no_row), kt7_path, no_row[:3]))
    if sorted(int(it["crop_id"]) for it in kt7_items) != list(range(kt7.n)):
        raise JudgeError("the KT7 items and %s disagree (crop ids)" % kt7_path)
    X_dino7, dino7_meta = _kt7_features(funnel_dir, kt7, DINO_KT7, dino_name,
                                        lambda: dinov2_embedder or E.Dinov2Embedder(model, pooling))
    X_s1, s1_info = adapter.step1_features()
    if len(X_s1) != crops.n:
        raise JudgeError("the Step 1 features have %d rows, the crop table %d" % (len(X_s1), crops.n))
    s1_name = s1_info.get("embedder")
    if not s1_name:
        raise JudgeError("the adapter's Step 1 features do not name their embedder")
    tag, factory = _adapter_embedder(adapter)
    X_s17, s17_meta = _kt7_features(funnel_dir, kt7, "emb_%s_kt7.npz" % tag, s1_name,
                                    (lambda: step1_embedder) if step1_embedder is not None else factory)
    # the adapter's Step 1 features have no one file to name: the record has no path
    # (check_records skips it) and its sha256 covers the crop table, the embedder and
    # the feature values themselves, so re-embedded features are a different input
    s1_record = {"path": None, "what": "adapter.step1_features",
                 "sha256": sha256_bytes(canonical_json({"crops_sha256": crops.sha, "embedder": s1_name,
                                                        "values_sha256": array_sha256(X_s1)}).encode("utf-8"))}

    # query keys
    unit_ids = crop_unit_ids(crops)
    keys = _unit_keys(adapter, unit_ids, domain, funnel_dir)
    session = {}
    for kt_items in known.values():
        for it in kt_items:
            if it.get("session"):
                session[it["id"]] = it["session"]
    q_crops = {kind: [(keys.get(u) or {}).get(kind) for u in unit_ids] for kind in DISJOINT_KEYS}
    q_crops["session"] = [session.get(u) for u in unit_ids]
    missing = [u for u in unit_ids if u not in keys]
    if missing:
        raise JudgeError("the adapter gave no keys for %d crop unit(s), e.g. %s" % (len(missing), missing[:3]))
    kt7_order = np.array([int(it["crop_id"]) for it in kt7_items], dtype=np.int64)
    kd7 = [item_keys(it, domain) for it in kt7_items]
    q_kt7 = {kind: [d.get(kind) for d in kd7] for kind in KEY_KINDS}

    q_digest = {"crops": keys_sha256(q_crops), "kt7": keys_sha256(q_kt7)}
    bnk = banks(domain, adapter, {"crops": X_dino, "kt7": X_dino7}, known_truth=known, crop_table=crops,
                funnel_dir=funnel_dir)
    modules = (sys.modules[__name__], E, adapter)
    common_inputs = {"crop_table": {"path": str(crops.path), "sha256": crops.sha},
                     "kt7_table": {"path": str(kt7.path), "sha256": kt7.sha}}
    known_digest = sha256_bytes(canonical_json(
        {k: [[it["id"], it.get("truth"), it.get("truth_kind"), it.get("role")] for it in sorted(v, key=lambda x: x["id"])]
         for k, v in sorted(known.items())}).encode("utf-8"))
    common_inputs["known_truth"] = {"path": None, "sha256": known_digest}
    index = {"files": {}}

    def meta_for(judge, set_name, labels, features, bank, exclusion, k, temperature, prompt_rows, extra):
        inputs = dict(common_inputs, features=features)
        m = header(SCORE_FORMAT, domain, prereg, inputs, seeds=(bank or {}).get("seeds") or {},
                   modules=modules, testing=testing)
        m.update({"judge": judge, "labels": labels, "set": set_name, "features": features,
                  "bank": (None if bank is None else {"kt": bank["kt"], "specs": bank["specs"], "n": bank["n"],
                                                      "sha256": bank["sha256"], "dropped": bank["dropped"],
                                                      "empty": bank["empty"]}),
                  "exclusion": exclusion, "k": k, "temperature": temperature, "prompts": prompt_rows,
                  "crops_sha256": kt7.sha if set_name == "kt7" else crops.sha})
        m.update(extra or {})
        return m

    for jid, entry in sorted(pnl.items()):
        kind = entry["kind"]
        if kind == "rl":
            continue                                   # scored from the labeller's answers (rl.py)
        if kind == "step1_probe":
            path = _score_path(funnel_dir, jid, "kt7")
            labels = [t["name"] for t in targets] + [OTHER]
            meta = meta_for(jid, "kt7", labels, _feature_record(s17_meta["path"], s17_meta["sha256"]), None,
                            "none", None, None, None, {"role": entry.get("role")})
            if not force and _would_noop(path, meta):
                index["files"]["%s__kt7" % jid] = {"path": str(path), "sha256": C.sha256_file(path)}
                continue
            Xq = np.asarray(X_s17[kt7_order], dtype=np.float32)
            ok = np.isfinite(Xq).all(axis=1)
            P = np.full((len(Xq), n_t + 1), np.nan, dtype=np.float32)
            cos = np.full((len(Xq), n_t + 1), np.nan, dtype=np.float32)
            if ok.any():
                Pj, cj = adapter.j1_scores(Xq[ok])
                Pj, cj = np.asarray(Pj, dtype=np.float32), np.asarray(cj, dtype=np.float32)
                if Pj.shape != (int(ok.sum()), n_t + 1) or cj.shape != Pj.shape:
                    raise JudgeError("j1_scores returned %s / %s for %d rows of %d labels"
                                     % (Pj.shape, cj.shape, int(ok.sum()), n_t + 1))
                P[ok], cos[ok] = Pj, cj
            rec = _write_scores(path, meta, kt7_order, P, {"cos": cos.astype(np.float16)}, force=force)
            index["files"]["%s__kt7" % jid] = {"path": rec["path"], "sha256": rec["sha256"]}
            continue
        if kind == "zero_shot":
            enc = text_encoder if text_encoder is not None else adapter.text_encoder()
            if resolver is None:
                from . import taxonomy
                cache = taxonomy.load_cache(funnel_dir / "taxonomy_cache.json", domain)
                resolver = taxonomy.Resolver(cache, domain)
            texts, groups, labels = prompts(domain, resolver)
            scale = getattr(enc, "logit_scale", None)
            if scale is None:
                raise JudgeError("the text tower has no logit_scale")
            T = text_features(enc, texts, X_s1.shape[1])
            prompt_rows = [{"text": t, "label": labels[g]} for t, g in zip(texts, groups)]
            enc_rec = {"name": getattr(enc, "name", type(enc).__name__), "scale": float(scale),
                       "provenance": getattr(enc, "provenance", None)}
            for set_name, Xs, feats, order in (
                    ("crops", X_s1, s1_record, np.arange(crops.n, dtype=np.int64)),
                    ("kt7", X_s17, _feature_record(s17_meta["path"], s17_meta["sha256"]), kt7_order)):
                path = _score_path(funnel_dir, jid, set_name)
                meta = meta_for(jid, set_name, labels, feats, None, "none", None, None, prompt_rows,
                                {"text_encoder": enc_rec})
                if not force and _would_noop(path, meta):
                    index["files"]["%s__%s" % (jid, set_name)] = {"path": str(path), "sha256": C.sha256_file(path)}
                    continue
                P = zero_shot(np.asarray(Xs, dtype=np.float32)[order], T, scale, groups, len(labels))
                rec = _write_scores(path, meta, order, P, force=force)
                index["files"]["%s__%s" % (jid, set_name)] = {"path": rec["path"], "sha256": rec["sha256"]}
            continue
        # knn
        b = bnk[jid]
        for set_name, Xs, feats, order, qk in (
                ("crops", X_dino, _shard_set_record(dino_info),
                 np.arange(crops.n, dtype=np.int64), q_crops),
                ("kt7", X_dino7, _feature_record(dino7_meta["path"], dino7_meta["sha256"]), kt7_order, q_kt7)):
            path = _score_path(funnel_dir, jid, set_name)
            # the query keys decide what each query may see: they are an input like the bank's
            meta = meta_for(jid, set_name, b["label_names"], feats, b, b["exclusion"], b["k"], b["temperature"],
                            None, {"query_keys_sha256": q_digest[set_name]})
            if not force and _would_noop(path, meta):
                index["files"]["%s__%s" % (jid, set_name)] = {"path": str(path), "sha256": C.sha256_file(path)}
                continue
            P = knn(np.asarray(Xs, dtype=np.float32)[order], qk, b["X"], b["labels"], b["keys"], b["n_labels"],
                    k=b["k"], temperature=b["temperature"], exclude=b["exclusion"])
            rec = _write_scores(path, meta, order, P, force=force)
            index["files"]["%s__%s" % (jid, set_name)] = {"path": rec["path"], "sha256": rec["sha256"]}
            log("%s on %s: %d rows" % (jid, set_name, len(order)))
    inputs = dict(common_inputs, dinov2_crops=_shard_set_record(dino_info),
                  dinov2_kt7=_feature_record(dino7_meta["path"], dino7_meta["sha256"]),
                  step1_crops=s1_record, step1_kt7=_feature_record(s17_meta["path"], s17_meta["sha256"]))
    out = header(INDEX_FORMAT, domain, prereg, inputs, modules=modules, testing=testing)
    out.update({"files": dict(sorted(index["files"].items())),
                "banks": {j: {"sha256": b["sha256"], "n": b["n"], "kt": b["kt"], "dropped": b["dropped"],
                              "seeds": b["seeds"], "empty": b["empty"]} for j, b in sorted(bnk.items())},
                "seconds": round(time.time() - t0, 1)})
    write_json_atomic(funnel_dir / "judges" / "index.json", out)
    log("scored %d file(s) in %.0fs" % (len(out["files"]), time.time() - t0))
    return out
