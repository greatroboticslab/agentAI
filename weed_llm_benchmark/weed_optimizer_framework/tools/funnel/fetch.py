"""Lab fetches (verb `fetch`, levers L11a and L12; contract §6 H12, §7 R-F, DEC-7,
DEC-8; runner §4.5, §5.2.5). The cluster's jobs have no network, so everything
the audit needs from the web is fetched here, hashed as received, written
under the funnel directory and listed in fetch_manifest.json, which the
cluster checks on arrival (check_manifest).

  taxonomy     the authority's answers for every source class name and every
               taxon the config names (taxonomy.build_cache) -> taxonomy_cache.json
  known-items  the known-item list of H12 (public datasets documented to hold
               a target class with boxes), compiled from the literature notes,
               the dataset surveys they cite and the index the config names,
               hashed before any registry comparison -> known_items_v1.json
  cards        every card resolver's cards, papers, class lists and upstream
               annotation archives -> cards/<slug>/..., cards/index.json
  kt7          independent species truth: research-grade observations with
               CC-licensed photos, at most per_taxon_max per taxon, each photo's
               observation id, licence and sha256 recorded -> kt7/
  refetch      (optional, R-F) the upstream images a harvest cap left out, as
               raw material for the whole Step 1 guard chain -> refetch/

Every HTTP call goes through transport(url, params) -> (status, bytes,
headers), injected by tests; the default uses urllib (60 s timeout, 3
attempts). A body is hashed before it is parsed, and a download whose provider
publishes a checksum (Mendeley sha256, Zenodo md5, Hugging Face LFS sha256)
must match it. API keys come from the environment and never reach a file.
Provider endpoints are constants here (they are not domain-specific); what to
fetch comes from the domain config. Nothing here names a domain.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from ..inc import common as C
from . import (FetchError, StaleInput, header, read_json, utc, write_json_atomic, _atomic_write_bytes,
               write_csv_atomic)
from . import domain as D
from .names import key as name_key

KINDS = ("http", "archive", "inat_observations", "roboflow_classes", "gbif_backbone")
PROVIDERS = ("url", "huggingface", "mendeley", "zenodo")
TIMEOUT = 60
ATTEMPTS = 3
USER_AGENT = "funnel-audit-fetch/1 (research data audit)"
HF_API = "https://huggingface.co/api/datasets/%s"
HF_FILE = "https://huggingface.co/datasets/%s/resolve/main/%s"
MENDELEY_DATASET = "https://data.mendeley.com/public-api/datasets/%s"
MENDELEY_FOLDERS = "https://data.mendeley.com/public-api/datasets/%s/folders/%s"
MENDELEY_FILES = "https://data.mendeley.com/public-api/datasets/%s/files"
ZENODO_RECORD = "https://zenodo.org/api/records/%s"
ROBOFLOW_PROJECT = "https://api.roboflow.com/%s/%s"
SECRET_PARAMS = ("api_key", "key", "token", "access_token")
MACHINE_LOCAL = ("kt7/crops_kt7.csv", "refetch/crops_refetch.csv")
MANIFEST_NAME = "fetch_manifest.json"
KT7_MAX_PAGES = 10


# ------------------------------------------------------------------ transport
def default_transport(url, params, headers=None):
    """GET url?params with urllib: (status, body bytes, headers). An HTTP error
    status is returned, not raised; a network failure is retried ATTEMPTS
    times, then raised as FetchError."""
    q = urllib.parse.urlencode(sorted((params or {}).items()))
    full = url + ("?" + q if q else "")
    req = urllib.request.Request(full, headers=dict({"User-Agent": USER_AGENT}, **(headers or {})))
    last = None
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.status, r.read(), dict(r.headers.items())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt + 1 < ATTEMPTS:
                last = e
                time.sleep(2 ** attempt)
                continue
            return e.code, e.read() or b"", dict(e.headers.items()) if e.headers else {}
        except (urllib.error.URLError, OSError) as e:
            last = e
            time.sleep(2 ** attempt)
    raise FetchError("GET %s failed after %d attempts (%s)" % (_redact(url, params), ATTEMPTS, last))


def _redact(url, params):
    p = {k: ("<redacted>" if k.lower() in SECRET_PARAMS else v) for k, v in sorted((params or {}).items())}
    q = urllib.parse.urlencode(sorted(p.items()))
    return url + ("?" + q if q else "")


def _get(transport, url, params=None):
    status, data, headers = (transport or default_transport)(url, params or {})
    data = data or b""
    return status, data, hashlib.sha256(data).hexdigest(), headers or {}


def _json(data, what):
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise FetchError("%s is not JSON (%s)" % (what, e))


# ------------------------------------------------------------------ prereg / header
def _prereg(prereg, out_dir):
    """The prereg the output header records: the one given, else the funnel
    directory's own prereg_v1.json (fetch runs with --out <funnel dir>)."""
    if prereg is not None:
        return prereg if hasattr(prereg, "core_sha256") else D.load_prereg(prereg)
    p = Path(out_dir) / "prereg_v1.json"
    if not p.is_file():
        raise FetchError("no prereg given and none at %s; the output header needs it" % p)
    return D.load_prereg(p)


def _self():
    import sys
    return sys.modules[__name__]


def _save(out_dir, rel, data):
    p = Path(out_dir) / rel
    sha = _atomic_write_bytes(p, data)
    return {"file": rel, "sha256": sha, "bytes": len(data)}


def _safe(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "x"


def _members(data):
    """[{"name", "sha256"}] of a zip archive's members, or None."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return None
    out = []
    for n in sorted(zf.namelist()):
        if n.endswith("/"):
            continue
        out.append({"name": n, "sha256": hashlib.sha256(zf.read(n)).hexdigest()})
    return out


# ------------------------------------------------------------------ providers
def _mendeley_listing(transport, dataset, version):
    """{folder path: [file record]} of a Mendeley dataset version, plus the raw
    listing bodies (hashed)."""
    raw = []
    st, data, sha, _h = _get(transport, MENDELEY_FOLDERS % (dataset, version))
    if st != 200:
        raise FetchError("Mendeley folders of %s v%s answered %s" % (dataset, version, st))
    raw.append({"url": MENDELEY_FOLDERS % (dataset, version), "sha256": sha})
    folders = _json(data, "Mendeley folders")
    by_id = {f["id"]: f for f in folders}

    def path_of(fid):
        parts = []
        while fid in by_id:
            parts.append(by_id[fid]["name"])
            fid = by_id[fid].get("parent_id")
        return "/".join(reversed(parts))
    out = {}
    for fid in ["root"] + sorted(by_id):
        st, data, sha, _h = _get(transport, MENDELEY_FILES % dataset, {"folder_id": fid, "version": str(version)})
        if st != 200:
            raise FetchError("Mendeley files of %s v%s folder %s answered %s" % (dataset, version, fid, st))
        raw.append({"url": MENDELEY_FILES % dataset, "params": {"folder_id": fid, "version": str(version)},
                    "sha256": sha})
        files = _json(data, "Mendeley files")
        out[path_of(fid) if fid != "root" else ""] = files
    return out, raw


def _select(names, spec):
    want = spec.get("files")
    rx = re.compile(spec["files_regex"]) if spec.get("files_regex") else None
    frx = re.compile(spec["folder_regex"]) if spec.get("folder_regex") else None
    out = []
    for folder, fname in names:
        if frx is not None and not frx.search(folder):
            continue
        if want is not None and fname not in want:
            continue
        if rx is not None and not rx.search(fname):
            continue
        out.append((folder, fname))
    return out


def _download_checked(transport, url, want_sha256=None, want_md5=None, params=None):
    st, data, sha, _h = _get(transport, url, params)
    if st != 200:
        raise FetchError("GET %s answered %s" % (_redact(url, params), st))
    if want_sha256 and sha != want_sha256:
        raise FetchError("%s: sha256 %s, the provider publishes %s" % (url, sha[:12], want_sha256[:12]))
    if want_md5 and hashlib.md5(data).hexdigest() != want_md5:
        raise FetchError("%s: md5 differs from the provider's %s" % (url, want_md5))
    return data, sha


def fetch_spec(spec, slug, cards_dir, transport):
    """Fetch one spec of a card resolver into cards_dir/<slug>/; returns index
    entries whose "file" is relative to cards_dir. HTTP failures of cards and
    papers are recorded (status), not raised; a checksum mismatch, a missing
    API key or a failed archive download raises FetchError."""
    kind, prov, what = spec.get("kind"), spec.get("provider", "url"), spec.get("what", "card")
    if kind not in KINDS:
        raise FetchError("%s: fetch kind %r is not one of %s" % (slug, kind, KINDS))
    out_dir = Path(cards_dir)
    base = _safe(slug)
    ents = []

    def entry(url, params, saved, status, licence=None, members=None, error=None):
        e = {"url": _redact(url, params), "kind": kind, "provider": prov, "what": what, "status": status,
             "licence": licence if licence is not None else spec.get("licence"), "members": members,
             "fetched_utc": utc(), "error": error}
        e.update(saved or {"file": None, "sha256": None, "bytes": 0})
        ents.append(e)
        return e
    if kind == "roboflow_classes":
        env = spec.get("api_key_env") or "ROBOFLOW_API_KEY"
        k = os.environ.get(env)
        url = ROBOFLOW_PROJECT % (spec["workspace"], spec["project"])
        if not k:
            raise FetchError("%s: the Roboflow class list needs an API key in $%s" % (slug, env))
        st, data, sha, _h = _get(transport, url, {"api_key": k})
        saved = _save(out_dir, "%s/roboflow_%s.json" % (base, _safe(spec["project"])), data) if st == 200 else None
        entry(url, {"api_key": k}, saved, st, error=None if st == 200 else "HTTP %s" % st)
        return ents
    if prov == "huggingface" and kind == "http":
        repo = spec["repo"]
        st, data, sha, _h = _get(transport, HF_API % repo)
        lic = None
        if st == 200:
            body = _json(data, "Hugging Face %s" % repo)
            lic = (body.get("cardData") or {}).get("license")
            entry(HF_API % repo, None, _save(out_dir, "%s/hf_%s.json" % (base, _safe(repo)), data), st, licence=lic)
        else:
            entry(HF_API % repo, None, None, st, error="HTTP %s" % st)
        st, data, sha, _h = _get(transport, HF_FILE % (repo, "README.md"))
        entry(HF_FILE % (repo, "README.md"), None,
              _save(out_dir, "%s/hf_%s_README.md" % (base, _safe(repo)), data) if st == 200 else None, st,
              licence=lic, error=None if st == 200 else "HTTP %s" % st)
        return ents
    if prov == "mendeley":
        ds, ver = spec["dataset"], spec["version"]
        st, data, sha, _h = _get(transport, MENDELEY_DATASET % ds, {"version": str(ver)})
        lic = None
        if st == 200:
            body = _json(data, "Mendeley %s" % ds)
            lic = (body.get("data_licence") or {}).get("short_name")
            if kind == "http":
                entry(MENDELEY_DATASET % ds, {"version": str(ver)},
                      _save(out_dir, "%s/mendeley_%s_v%s.json" % (base, ds, ver), data), st, licence=lic)
        elif kind == "http":
            entry(MENDELEY_DATASET % ds, {"version": str(ver)}, None, st, error="HTTP %s" % st)
        if kind == "http":
            return ents
        listing, raw = _mendeley_listing(transport, ds, ver)
        _save(out_dir, "%s/mendeley_%s_v%s_files.json" % (base, ds, ver),
              json.dumps({"listing": listing, "requests": raw}, sort_keys=True, indent=1).encode("utf-8"))
        rec = {(folder, f["filename"]): f for folder, files in listing.items() for f in files}
        chosen = _select(sorted(rec), spec)
        if not chosen:
            raise FetchError("%s: no Mendeley file of %s v%s matches %s" % (slug, ds, ver, spec))
        single = spec.get("files") is not None and len(chosen) == 1
        for folder, fname in chosen:
            f = rec[(folder, fname)]
            cd = f.get("content_details") or {}
            data, sha = _download_checked(transport, cd["download_url"], want_sha256=cd.get("sha256_hash"))
            rel = ("%s/%s" % (base, _safe(fname)) if single else
                   "%s/annotations/%s%s" % (base, "/".join(_safe(p) for p in folder.split("/") if p) + "/"
                                            if folder else "", _safe(fname)))
            entry(cd["download_url"], None, _save(out_dir, rel, data), 200, licence=lic,
                  members=_members(data) if fname.lower().endswith(".zip") else None)
        return ents
    if prov == "zenodo":
        rid = spec["record"]
        st, data, sha, _h = _get(transport, ZENODO_RECORD % rid)
        if st != 200:
            entry(ZENODO_RECORD % rid, None, None, st, error="HTTP %s" % st)
            return ents
        body = _json(data, "Zenodo %s" % rid)
        lic = ((body.get("metadata") or {}).get("license") or {}).get("id")
        entry(ZENODO_RECORD % rid, None, _save(out_dir, "%s/zenodo_%s.json" % (base, rid), data), st, licence=lic)
        if kind == "archive":
            files = {f["key"]: f for f in body.get("files") or []}
            for (_fo, fname) in _select([("", k) for k in sorted(files)], spec):
                f = files[fname]
                ck = str(f.get("checksum") or "")
                data, sha = _download_checked(transport, f["links"]["self"],
                                              want_md5=ck[4:] if ck.startswith("md5:") else None)
                entry(f["links"]["self"], None, _save(out_dir, "%s/%s" % (base, _safe(fname)), data), 200,
                      licence=lic, members=_members(data) if fname.lower().endswith(".zip") else None)
        return ents
    if prov == "url":
        url = spec["url"]
        st, data, sha, _h = _get(transport, url, spec.get("params"))
        ext = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".html"
        name = "%s/%s_%s%s" % (base, what, hashlib.sha256(url.encode("utf-8")).hexdigest()[:10], ext)
        entry(url, spec.get("params"), _save(out_dir, name, data) if st == 200 else None, st,
              error=None if st == 200 else "HTTP %s" % st,
              members=_members(data) if st == 200 and ext.lower() == ".zip" else None)
        return ents
    raise FetchError("%s: provider %r with kind %r is not implemented (providers %s)" % (slug, prov, kind, PROVIDERS))


def fetch_cards(domain, out_dir, transport, prereg=None, testing=False):
    """L11a: every card resolver's fetch specs and upstream annotations, then
    cards/index.json. A card or paper that answers an HTTP error is recorded;
    a refusal (missing API key) is recorded as refused and listed."""
    dom = D.load(domain)
    pre = _prereg(prereg, out_dir)
    res_all = (dom.raw.get("sources") or {}).get("card_resolvers") or {}
    cards, refused = {}, []
    for slug in sorted(res_all):
        res = res_all[slug] or {}
        specs = list(res.get("fetch") or [])
        if res.get("upstream_annotations"):
            specs.append(res["upstream_annotations"])
        ents = []
        for spec in specs:
            try:
                ents.extend(fetch_spec(spec, slug, Path(out_dir) / "cards", transport))
            except FetchError as e:
                if spec.get("kind") == "roboflow_classes":
                    refused.append({"source": slug, "why": str(e)})
                    ents.append({"url": None, "kind": spec.get("kind"), "provider": spec.get("provider"),
                                 "what": spec.get("what"), "status": "refused", "file": None, "sha256": None,
                                 "bytes": 0, "licence": None, "members": None, "fetched_utc": utc(),
                                 "error": str(e)})
                    continue
                raise
        cards[slug] = ents
    doc = header("cards", dom, pre, {}, modules=(_self(),), testing=testing)
    doc.update({"cards": cards, "refused": refused})
    write_json_atomic(Path(out_dir) / "cards" / "index.json", doc)
    return doc


# ------------------------------------------------------------------ KT7
def kt7_roles(items, domain):
    """Roles of the independent-truth photos (runner §4.6): per taxon, sorted by
    id and permuted with seed funnel/v1/kt7/roles/<taxon>; the first
    roles.exemplar are exemplars; of the rest, position i is g0 when
    i % g0_every == g0_every - 1, else sentinel."""
    import numpy as np
    dom = D.load(domain)
    roles = ((dom.raw.get("known_truth") or {}).get("kt7") or {}).get("roles") or {}
    n_ex, every = int(roles.get("exemplar", 0)), int(roles.get("g0_every", 0))
    by = {}
    for it in items:
        by.setdefault(it["taxon"], []).append(dict(it))
    out = []
    for taxon in sorted(by):
        rows = sorted(by[taxon], key=lambda r: r["id"])
        perm = np.random.default_rng(C.stable_int("funnel/v1/kt7/roles/%s" % taxon)).permutation(len(rows))
        for pos, idx in enumerate(perm.tolist()):
            r = rows[idx]
            if pos < n_ex:
                r["role"] = "exemplar"
            else:
                i = pos - n_ex
                r["role"] = "g0" if every > 0 and i % every == every - 1 else "sentinel"
            out.append(r)
    out.sort(key=lambda r: r["id"])
    return out


def _photo_url(url, size):
    return re.sub(r"/square(\.[A-Za-z0-9]+)(\?.*)?$", r"/%s\1" % size, url)


def fetch_kt7(domain, out_dir, transport, prereg=None, testing=False):
    """DEC-8: at most per_taxon_max observations per taxon, one CC-licensed
    photo each (the first whose licence is allowed), from the provider query
    the config names verbatim. Writes kt7/photos/, kt7/responses/,
    kt7/kt7_items.jsonl and kt7/crops_kt7.csv."""
    dom = D.load(domain)
    pre = _prereg(prereg, out_dir)
    cfg = (dom.raw.get("known_truth") or {}).get("kt7") or {}
    prov = cfg.get("provider") or {}
    if prov.get("kind") != "inat_observations":
        raise FetchError("known_truth.kt7.provider.kind is %r, not inat_observations" % prov.get("kind"))
    maxn = int(cfg["per_taxon_max"])
    allowed = set(cfg.get("licences") or [])
    if not allowed:
        raise FetchError("known_truth.kt7.licences is empty; DEC-8 takes CC-licensed photos only")
    size = prov.get("photo_size", "large")
    targets = {t["taxon"]: t["name"] for t in dom.targets}
    atts = {a["taxon"]: a["id"] for a in dom.attractors}
    items, per_taxon, skipped = [], {}, {"licence": 0, "grade": 0, "no_photo": 0, "taxon": 0}
    out_dir = Path(out_dir)
    for taxon in cfg.get("taxa") or []:
        got = []
        seen = set()
        for page in range(1, KT7_MAX_PAGES + 1):
            params = dict(prov.get("params") or {})
            params.update({"taxon_name": taxon, "page": str(page)})
            params.setdefault("per_page", str(maxn))
            st, data, sha, _h = _get(transport, prov["url"], params)
            if st != 200:
                raise FetchError("%s for %r answered %s" % (prov["url"], taxon, st))
            _save(out_dir, "kt7/responses/%s_p%02d.json" % (_safe(name_key(taxon)), page), data)
            body = _json(data, "observations of %r" % taxon)
            res = body.get("results") or []
            for ob in res:
                if len(got) >= maxn:
                    break
                oid = ob.get("id")
                if oid in seen:
                    continue
                seen.add(oid)
                if ob.get("quality_grade") != "research":
                    skipped["grade"] += 1
                    continue
                # the truth is the queried taxon, so the observation must be of it or of a
                # descendant (a name query can match other taxa: a synonym, a homonym)
                observed = str((ob.get("taxon") or {}).get("name") or "")
                if not (observed == taxon or observed.startswith(taxon + " ")):
                    skipped["taxon"] += 1
                    continue
                photo = next((p for p in ob.get("photos") or [] if p.get("license_code") in allowed), None)
                if photo is None:
                    skipped["licence" if ob.get("photos") else "no_photo"] += 1
                    continue
                url = _photo_url(photo["url"], size)
                pdata, psha = _download_checked(transport, url)
                ext = os.path.splitext(urllib.parse.urlparse(url).path)[1] or ".jpg"
                fname = "%s_%s%s" % (oid, photo["id"], ext.lower())
                _save(out_dir, "kt7/photos/%s" % fname, pdata)
                got.append({"id": "t7:%s/%s" % (oid, photo["id"]), "taxon": taxon, "target": targets.get(taxon),
                            "attractor": atts.get(taxon), "observation_id": oid, "photo_id": photo["id"],
                            "url": url, "licence": photo.get("license_code"), "sha256": psha,
                            "quality_grade": ob.get("quality_grade"), "observed_on": ob.get("observed_on"),
                            "place": ob.get("place_guess"), "observed_taxon": observed,
                            "query": {k: v for k, v in sorted(params.items())},
                            "file": fname, "role": None})
            if len(got) >= maxn or len(res) < int(params.get("per_page", maxn)):
                break
        per_taxon[taxon] = len(got)
        items.extend(got)
    items = kt7_roles(items, dom)
    rec = _save(out_dir, "kt7/kt7_items.jsonl",
                "".join(json.dumps(r, sort_keys=True) + "\n" for r in items).encode("utf-8"))
    table = kt7_table(out_dir, dom)
    summary = {"items": len(items), "per_taxon": per_taxon, "skipped": skipped, "per_taxon_max": maxn,
               "table": table}
    doc = header("funnel-kt7-index/1", dom, pre, {"kt7_items": {"path": str(out_dir / rec["file"]),
                                                                "sha256": rec["sha256"], "bytes": rec["bytes"]}},
                 modules=(_self(),), testing=testing)
    doc.update(dict(summary, provider={"kind": prov["kind"], "url": prov["url"], "params": prov.get("params"),
                                        "photo_size": size}, licences=sorted(allowed),
                    rule="one CC-licensed photo (the first allowed) per research-grade observation whose own "
                         "taxon is the queried taxon or one below it, at most per_taxon_max observations per "
                         "taxon, in the provider's order"))
    write_json_atomic(out_dir / "kt7" / "kt7_index.json", doc)
    return summary


def kt7_table(out_dir, domain):
    """kt7/crops_kt7.csv (verify's CROP_FIELDS, set "kt7"), one row per photo in
    id order, the box covering the whole photo; image paths are this machine's
    (the file is machine-local and rebuilt on arrival by check_manifest)."""
    from PIL import Image, ImageOps
    from ..inc import verify as V
    dom = D.load(domain)
    out_dir = Path(out_dir)
    p = out_dir / "kt7" / "kt7_items.jsonl"
    rows = [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows.sort(key=lambda r: r["id"])
    other = dom.other["id"]
    body = []
    for i, r in enumerate(rows):
        img = out_dir / "kt7" / "photos" / r["file"]
        with Image.open(img) as im0:
            W, H = ImageOps.exif_transpose(im0).size
        label = dom.class_id(r["target"]) if r.get("target") else other
        body.append([i, "kt7", r["id"], str(img.resolve()), "kt7", str(r["observation_id"]), 0,
                     "0.500000", "0.500000", "1.000000", "1.000000", W, H, label, r["taxon"]])
    sha = write_csv_atomic(out_dir / "kt7" / "crops_kt7.csv", V.CROP_FIELDS, body)
    return {"path": str(out_dir / "kt7" / "crops_kt7.csv"), "sha256": sha, "rows": len(body)}


# ------------------------------------------------------------------ taxonomy
def taxonomy(domain, names_from, out_path, transport, prereg=None, testing=False):
    """L12: resolve every source class name of names_from (a pool summary: the
    names of per_slug[*].join; or a JSON list of names) and every taxon the
    config names; writes taxonomy_cache.json."""
    from . import taxonomy as T
    dom = D.load(domain)
    pre = _prereg(prereg, Path(out_path).parent)
    src = read_json(names_from)
    names = []
    if isinstance(src, list):
        names = [str(n) for n in src]
    elif isinstance(src, dict) and isinstance(src.get("per_slug"), dict):
        for st in src["per_slug"].values():
            for v in (st.get("join") or {}).values():
                if v and v[0]:
                    names.append(str(v[0]))
    else:
        raise FetchError("%s holds neither a name list nor per_slug joins" % names_from)
    return T.build_cache(sorted(set(names)), dom, transport or default_transport, out_path, prereg=pre,
                         inputs={"names_from": names_from}, testing=testing)


# ------------------------------------------------------------------ H12 known items
_WS = re.compile(r"[^a-z0-9]+")


def _norm_text(t):
    return " %s " % _WS.sub(" ", str(t or "").lower()).strip()


def _camel(name):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(name))


def _target_phrases(dom):
    out = {}
    for t in dom.targets:
        ph = {_norm_text(_camel(t["name"])), _norm_text(t.get("common") or ""), _norm_text(t.get("taxon") or "")}
        out[t["name"]] = sorted(p for p in ph if p.strip())
    return out


def _named(text, phrases):
    nt = _norm_text(text)
    return sorted(n for n, ps in phrases.items() if any(p in nt for p in ps))


def _has_boxes(text, words):
    """Box words found at a word start (a plural or suffix may follow)."""
    nt = _norm_text(text)
    return sorted(w for w in words if _norm_text(w).rstrip() in nt)


def _key_pattern(k):
    """A regex matching a slug whose letters and digits contain k."""
    return "(?i)" + "[^a-z0-9]*".join(re.escape(c) for c in k)


def _slug_patterns(urls, templates):
    pats = []
    for u in urls:
        for t in templates:
            m = re.search(t["url_regex"], u)
            if m:
                slug = t["slug"].format(**{k: v for k, v in m.groupdict().items()})
                pats.append("^%s$" % re.escape(slug))
    return sorted(set(pats))


_LINK = re.compile(r"\[\[?([^\]]*)\]?\]\((https?://[^)\s]+)\)")


def _survey_entries(text):
    """Dataset entries of a survey README: blank-line separated paragraphs that
    hold a [dataset] link."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        links = _LINK.findall(para)
        ds = [u for lab, u in links if "dataset" in lab.lower() or "data" == lab.lower()]
        if ds:
            out.append({"text": para.strip(), "dataset_urls": ds, "urls": [u for _l, u in links]})
    return out


def _note(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    title = ""
    meta = {}
    for ln in text.splitlines():
        if ln.startswith("# ") and not title:
            title = ln[2:].strip()
        m = re.match(r"^(id|bib|topics|status):\s*(.*)$", ln)
        if m:
            meta[m.group(1)] = m.group(2).strip()
    return title, meta, text


class _Items(object):
    """The known items found so far, merged by name key."""

    def __init__(self, phrases, cfg):
        self.phrases, self.cfg, self.items = phrases, cfg, {}

    def add(self, name, url, doc_ref, text, urls, boxes=None, extra_patterns=(), name_pattern=True, title=None):
        """Admit a candidate when its text names a target and shows boxes
        (box words, or the index's own detection task given as boxes)."""
        named = _named(text, self.phrases)
        boxes = _has_boxes(text, self.cfg["box_words"]) if boxes is None else list(boxes)
        if not named or not boxes:
            return None
        k = name_key(name)
        pats = set(_slug_patterns(urls, self.cfg["slug_templates"])) | set(extra_patterns)
        if name_pattern and len(k) >= 4:
            pats.add(_key_pattern(k))
        it = self.items.get(k)
        if it is None:
            it = self.items[k] = {"name": name, "url": url, "doc_ref": doc_ref, "species_named": named,
                                  "has_boxes": True, "box_evidence": boxes, "slug_patterns": sorted(pats)}
            if title:
                it["title"] = title
        else:
            it["species_named"] = sorted(set(it["species_named"]) | set(named))
            it["slug_patterns"] = sorted(set(it["slug_patterns"]) | pats)
            it["box_evidence"] = sorted(set(it["box_evidence"]) | set(boxes))
            it["doc_ref"] = it["doc_ref"] + ";" + doc_ref
        return it


def known_items(sources, out_path, domain=None, transport=None, prereg=None, testing=False):
    """H12: the known-item list, from the literature notes (a directory of notes
    in the corpus format, or single note files), the dataset surveys those
    notes cite (fetched through the transport and hashed), the indexes the
    config names (sources.known_items.indexes) and any index (.json) or survey
    (.md without note metadata) file given directly. The rule is the
    config's (sources.known_items.rule); every item records the evidence that
    admitted it. Written before any registry comparison (census compares)."""
    out_path = Path(out_path)
    pre = _prereg(prereg, out_path.parent)
    dom = D.load(domain if domain is not None else pre.domain_name)
    cfg = (dom.raw.get("sources") or {}).get("known_items") or {}
    for k in ("box_words", "slug_templates", "rule"):
        if k not in cfg:
            raise FetchError("domain %s: sources.known_items.%s is required for H12" % (dom.name, k))
    found = _Items(_target_phrases(dom), cfg)
    topics = set(cfg.get("survey_topics") or [])
    survey_words = [w.lower() for w in cfg.get("survey_title_words") or ["survey"]]
    used, surveys, files = [], [], []
    side = out_path.parent / "known_items_sources"
    for s in sources:
        s = Path(s)
        if s.is_dir():
            files.extend(sorted(p for p in s.iterdir() if p.suffix == ".md" and p.name.lower() != "readme.md"))
        elif s.is_file():
            files.append(s)
        else:
            raise FetchError("known-items source %s does not exist" % s)
    for f in files:
        data = f.read_bytes()
        used.append({"ref": str(f), "sha256": hashlib.sha256(data).hexdigest()})
        if f.suffix == ".json":
            _index_items(_json(data, str(f)), {"format": "task_class_index"}, str(f), found)
            continue
        title, meta, text = _note(f)
        if not meta:                                   # a survey README given directly
            surveys.append((str(f), text))
            continue
        if topics and not (set(t.strip() for t in meta.get("topics", "").split(",")) & topics):
            continue
        urls = re.findall(r"https?://[^\s)>\]]+", text)
        if any(w in title.lower() for w in survey_words):
            for u in urls:
                raw = _github_readme(u)
                if raw is None:
                    continue
                st, body, sha, _h = _get(transport, raw)
                if st == 200:
                    rel = "%s.md" % _safe(u.split("github.com/", 1)[1])
                    _atomic_write_bytes(side / rel, body)
                    used.append({"ref": raw, "sha256": sha, "file": str(side / rel)})
                    surveys.append((raw, body.decode("utf-8", "replace")))
                else:
                    used.append({"ref": raw, "status": st})
            continue
        m = re.search(r"\(([^()]+)\)\s*$", title)
        short = m.group(1) if m else title
        if "dataset" in text.lower():
            found.add(short, (meta.get("bib") or "").split(" ")[-1], "note:%s" % meta.get("id", f.name), text, urls)
    for ref, text in surveys:
        for e in _survey_entries(text):
            u = urllib.parse.urlparse(e["dataset_urls"][0])
            found.add((u.netloc + u.path).rstrip("/"), e["dataset_urls"][0], "survey:%s" % ref, e["text"],
                      e["dataset_urls"], name_pattern=False, title=e["text"].splitlines()[0][:200])
    for spec in cfg.get("indexes") or []:
        st, data, sha, _h = _get(transport, spec["url"])
        if st != 200:
            raise FetchError("index %s answered %s" % (spec["url"], st))
        rel = _safe(urllib.parse.urlparse(spec["url"]).path.split("/")[-1] or "index")
        _atomic_write_bytes(side / rel, data)
        used.append({"ref": spec["url"], "sha256": sha, "file": str(side / rel)})
        _index_items(_json(data, spec["url"]), spec, spec["url"], found)
    doc = header("known_items", dom, pre, {}, modules=(_self(),), testing=testing)
    doc.update({"rule": cfg["rule"], "sources_used": used,
                "items": [found.items[k] for k in sorted(found.items)]})
    write_json_atomic(out_path, doc)
    return doc


def _github_readme(url):
    m = re.match(r"https?://github\.com/([^/\s]+)/([^/\s#?]+)", url)
    if not m:
        return None
    return "https://raw.githubusercontent.com/%s/%s/main/README.md" % (m.group(1), m.group(2))


def _index_items(body, spec, ref, found):
    """Items of a dataset index. Format task_class_index: {name: {ml_task,
    classes, docs_url, ...}}; an entry with a detection task shows boxes, and
    the index's slug template gives the registry slug it would have."""
    if spec.get("format") != "task_class_index":
        raise FetchError("index format %r is not implemented" % spec.get("format"))
    if not isinstance(body, dict):
        raise FetchError("index %s is not an object of entries" % ref)
    tasks = set(found.cfg.get("detection_tasks") or ["object_detection"])
    tmpl = spec.get("slug_template")
    for name in sorted(body):
        info = body[name] or {}
        if info.get("ml_task") not in tasks:
            continue
        classes = info.get("classes") or {}
        vals = classes.values() if isinstance(classes, dict) else classes
        text = " ".join([name.replace("_", " ")] + [str(v).replace("_", " ") for v in vals])
        url = info.get("docs_url") or ""
        extra = ["^%s$" % re.escape(tmpl.format(name=name))] if tmpl else []
        found.add(name, url, "index:%s" % ref, text, [url], boxes=["ml_task=%s" % info.get("ml_task")],
                  extra_patterns=extra)


# ------------------------------------------------------------------ R-F refetch
def refetch(domain, slug, out_dir, transport, prereg=None, testing=False):
    """R-F (optional): the upstream images of `slug` that the pool's copy of the
    source left out (upstream annotation files that geometry matched to no pool
    image, relation_geometry_v1.json), from the archive the card resolver's
    refetch_images spec names. Writes refetch/images/<slug>/, the upstream
    labels in upstream ids (refetch/labels/<slug>/), refetch/manifest.json and
    refetch/crops_refetch.csv. They are raw material only: the whole Step 1
    guard chain runs on them on the cluster before anything else (recover
    refuses R-F without that chain's record)."""
    from . import relation as REL
    dom = D.load(domain)
    pre = _prereg(prereg, out_dir)
    out_dir = Path(out_dir)
    res = ((dom.raw.get("sources") or {}).get("card_resolvers") or {}).get(slug) or {}
    spec = res.get("refetch_images")
    up_spec = res.get("upstream_annotations")
    if not spec or not up_spec:
        raise FetchError("%s: the card resolver names no refetch_images or upstream_annotations spec" % slug)
    geo_p = out_dir / "relation_geometry_v1.json"
    if not geo_p.exists():
        raise FetchError("refetch needs relation_geometry_v1.json (map --part geometry) to know which images "
                         "the pool lacks")
    geo = read_json(geo_p)
    mt = (geo.get("matches") or {}).get(slug)
    if not mt or "paired_upstream" not in mt:
        raise FetchError("relation_geometry_v1.json holds no paired upstream files for %s" % slug)
    idx = read_json(out_dir / "cards" / "index.json")
    ann = [e for e in (idx.get("cards") or {}).get(slug, []) if e.get("what") == "annotations" and e.get("file")]
    if len(ann) != 1:
        raise FetchError("%s: expected one fetched annotation archive, found %d" % (slug, len(ann)))
    up = REL.read_upstream(out_dir / "cards" / ann[0]["file"], up_spec.get("format", "yolo"),
                           classes_file=up_spec.get("classes_file"), voc_dir=up_spec.get("voc_dir"),
                           yolo_dir=up_spec.get("yolo_dir"))
    missing = sorted(set(up) - set(mt["paired_upstream"]))
    ents = fetch_spec(dict(spec, what="images"), slug, out_dir / "refetch" / "_archive", transport)
    # cards_dir here is refetch/_archive: entry files are relative to it
    arch = [e for e in ents if e.get("file")]
    if len(arch) != 1:
        raise FetchError("%s: the refetch_images spec must name exactly one archive" % slug)
    zp = out_dir / "refetch" / "_archive" / arch[0]["file"]
    zf = zipfile.ZipFile(zp)
    by_stem = {}
    for n in zf.namelist():
        st = os.path.splitext(n.split("/")[-1])[0]
        if st and not n.endswith("/") and os.path.splitext(n)[1].lower() in C.IMG_EXTS:
            by_stem.setdefault(st, n)
    rows = []
    lic = arch[0].get("licence")
    for stem in missing:
        member = by_stem.get(stem)
        if member is None:
            continue
        data = zf.read(member)
        ext = os.path.splitext(member)[1].lower()
        img_rel = "refetch/images/%s/%s%s" % (_safe(slug), _safe(stem), ext)
        rec = _save(out_dir, img_rel, data)
        boxes = up[stem]
        lines = []
        for (cid, x0, y0, x1, y1, W, H) in boxes:
            if W and H:
                x0, x1, y0, y1 = x0 / W, x1 / W, y0 / H, y1 / H
            lines.append("%d %.6f %.6f %.6f %.6f\n" % (int(cid), (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
        lab_rel = "refetch/labels/%s/%s.txt" % (_safe(slug), _safe(stem))
        lrec = _save(out_dir, lab_rel, "".join(lines).encode("utf-8"))
        rows.append({"stem": stem, "image": img_rel, "sha256": rec["sha256"], "label": lab_rel,
                     "label_sha256": lrec["sha256"], "boxes": len(boxes), "licence": lic})
    doc = header("refetch_manifest", dom, pre, {"relation_geometry": geo_p}, modules=(_self(),), testing=testing)
    doc.update({"slug": slug, "archive": {"file": str(zp), "sha256": arch[0]["sha256"]}, "licence": lic,
                "missing_upstream": len(missing), "images": rows,
                "chain": "not run: the never-train guard, the reference-copy check, exact_dup, H6 detect and the "
                         "embedding run on the cluster before recovery"})
    write_json_atomic(out_dir / "refetch" / "manifest.json", doc)
    refetch_table(out_dir)
    return {"images": len(rows), "missing_upstream": len(missing)}


def refetch_table(out_dir):
    """refetch/crops_refetch.csv (CROP_FIELDS, set "refetch", label = the
    upstream class id), machine-local like the KT7 table."""
    from PIL import Image, ImageOps
    from ..inc import verify as V
    out_dir = Path(out_dir)
    man = read_json(out_dir / "refetch" / "manifest.json")
    body = []
    for r in sorted(man["images"], key=lambda r: r["stem"]):
        img = out_dir / r["image"]
        with Image.open(img) as im0:
            W, H = ImageOps.exif_transpose(im0).size
        for b, ln in enumerate((out_dir / r["label"]).read_text().splitlines()):
            t = ln.split()
            body.append([len(body), "refetch", r["stem"], str(img.resolve()), man["slug"], man["slug"], b,
                         t[1], t[2], t[3], t[4], W, H, int(t[0]), t[0]])
    sha = write_csv_atomic(out_dir / "refetch" / "crops_refetch.csv", V.CROP_FIELDS, body)
    return {"sha256": sha, "rows": len(body)}


# ------------------------------------------------------------------ manifest
_FETCH_TOP = ("taxonomy_cache.json", "known_items_v1.json")
_FETCH_DIRS = ("cards", "kt7", "refetch", "known_items_sources")


def _fetched_files(out_dir):
    out_dir = Path(out_dir)
    files = []
    for n in _FETCH_TOP:
        if (out_dir / n).is_file():
            files.append(n)
    for d in _FETCH_DIRS:
        base = out_dir / d
        if base.is_dir():
            for p in sorted(base.rglob("*")):
                if p.is_file() and not p.name.endswith(".tmp"):
                    rel = p.relative_to(out_dir).as_posix()
                    if rel not in MACHINE_LOCAL:
                        files.append(rel)
    return sorted(files)


def write_manifest(out_dir, prereg=None, testing=False):
    """fetch_manifest.json: {relpath: sha256} of everything fetch wrote (the
    machine-local crop tables excepted: check_manifest rebuilds them)."""
    out_dir = Path(out_dir)
    dom_pre = _prereg(prereg, out_dir)
    dom = D.load(dom_pre.domain_name)
    files = {rel: C.sha256_file(out_dir / rel) for rel in _fetched_files(out_dir)}
    doc = header("fetch_manifest", dom, dom_pre, {}, modules=(_self(),), testing=testing)
    doc.update({"files": files, "machine_local": list(MACHINE_LOCAL)})
    write_json_atomic(out_dir / MANIFEST_NAME, doc)
    return {"files": len(files)}


def check_manifest(out_dir, rebuild_local=True):
    """On arrival (cluster): every file fetch_manifest.json lists must hash as
    recorded (StaleInput naming the first that does not). Then the
    machine-local crop tables are rebuilt with this machine's paths."""
    out_dir = Path(out_dir)
    man = read_json(out_dir / MANIFEST_NAME)
    for rel, sha in sorted((man.get("files") or {}).items()):
        p = out_dir / rel
        if not p.is_file():
            raise StaleInput("fetched file %s is missing" % rel)
        got = C.sha256_file(p)
        if got != sha:
            raise StaleInput("fetched file %s changed in transit (%s, recorded %s)" % (rel, got[:12], sha[:12]))
    if rebuild_local:
        if (out_dir / "kt7" / "kt7_items.jsonl").is_file():
            kt7_table(out_dir, man["domain"])
        if (out_dir / "refetch" / "manifest.json").is_file():
            refetch_table(out_dir)
