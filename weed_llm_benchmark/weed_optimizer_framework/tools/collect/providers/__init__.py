"""Providers of the collector (docs/CONTINUOUS_LOOP.md §7.2).

A provider turns a repository's API into four calls:
  search(query, limit)   candidate records a query finds ([] when the
                         repository has no search the collector can use);
  describe(ref)          one candidate record, with its licence, declared
                         classes (with box counts where the repository gives
                         them), annotation kind, size and image count;
  files(meta, select)    the downloadable files: name, url or path, size and
                         the checksum the provider publishes;
  fetch_file(spec, dest, max_bytes)
                         one file streamed into staging, capped and checked
                         (transport.Net).
and probe() (reachability, lever R0 network probe) and credentials().

A candidate record (meta) holds: provider, ref, version, source_id, title,
url, description, keywords, licence_text, licence_evidence, classes
([{"id", "name", "hints", "boxes", "images"}] or None), classes_source
("declared", "none"), annotation ("boxes", "image_level", "unknown") and its
evidence, bytes, images, found_by and raw_sha256 (the sha256 of each API
answer it was read from).

The providers: zenodo, mendeley, huggingface, kaggle, github, roboflow,
weedai (an annotation index with per-image category documents) and mediatum
(a record API plus an FTP server). Endpoints of repositories whose host names
would name a domain come from the config (base_url); the rest default here and
the config may override them (providers.<name>.api). Nothing here names a
domain.
"""
from __future__ import annotations

import html
import importlib
import re

from .. import ConfigError, ProviderError, safe_name
from ..transport import read_secret

KINDS = ("zenodo", "mendeley", "huggingface", "kaggle", "github", "roboflow", "weedai", "mediatum")
DEFAULT_DATA_REGEX = r"(?i)\.(zip|tar|tgz|tar\.gz|tar\.xz|7z|jpe?g|png|bmp|tiff?|json|xml|csv|txt|ya?ml|parquet)$"
DEFAULT_SKIP_REGEX = r"(?i)(\.(pdf|docx?|pptx?|ipynb|py|m|r|mp4|avi|mov|h5|pt|pth|onnx|ckpt|weights)$|(^|/)\.)"


def strip_html(text, limit=4000):
    t = re.sub(r"<[^>]+>", " ", str(text or ""))
    t = html.unescape(re.sub(r"\s+", " ", t)).strip()
    return t[:limit]


def slug_part(s):
    return re.sub(r"[^a-z0-9_]+", "_", str(s).lower()).strip("_")


class Provider(object):
    kind = None
    searchable = True

    def __init__(self, name, sec, net, cfg):
        self.name = name
        self.sec = dict(sec or {})
        self.net = net
        self.cfg = cfg

    # ---- configuration
    def api(self, key, default):
        return (self.sec.get("api") or {}).get(key, default)

    def credentials(self):
        """(ok, detail): whether the credential the provider needs is present."""
        cr = self.sec.get("credentials") or {}
        if not cr or cr.get("optional"):
            return True, None
        v, where = read_secret(cr.get("env") or (), cr.get("files") or ())
        if v:
            return True, where
        return False, "provider %s needs a credential in %s (card X16)" % (
            self.name, ", ".join(["$%s" % e for e in cr.get("env") or ()] + list(cr.get("files") or ())))

    def secret(self):
        cr = self.sec.get("credentials") or {}
        v, _w = read_secret(cr.get("env") or (), cr.get("files") or ())
        return v

    # ---- records
    def meta(self, ref, **kw):
        m = {"provider": self.name, "kind": self.kind, "ref": str(ref), "version": None,
             "source_id": self.source_id(ref), "title": None, "url": None, "description": None,
             "keywords": [], "licence_text": None, "licence_evidence": None, "classes": None,
             "classes_source": "none", "annotation": "unknown", "annotation_evidence": [], "bytes": None,
             "images": None, "found_by": [], "raw_sha256": []}
        m.update(kw)
        return m

    def source_id(self, ref):
        return "%s_%s" % (self.name, safe_name(ref))

    # ---- calls (subclasses)
    def search(self, query, limit):
        return []

    def describe(self, ref):
        raise NotImplementedError

    def files(self, meta, select=None):
        raise NotImplementedError

    def fetch_file(self, spec, dest, max_bytes=None, select=None):
        return self.net.download(spec["url"], dest, params=spec.get("params"), headers=self._headers(spec),
                                 max_bytes=max_bytes, expected_size=spec.get("size"),
                                 checksums=spec.get("checksums"), what="%s %s" % (self.name, spec["name"]))

    def _headers(self, spec):
        return None

    def probe(self):
        url = self.sec.get("probe_url") or self.default_probe_url()
        if not url:
            return {"reachable": None, "why": "no probe url"}
        import time
        t0 = time.time()
        try:
            st, _raw, _h = self.net.request(url)
        except ProviderError as e:
            return {"reachable": False, "url": url, "error": str(e)[:300], "seconds": round(time.time() - t0, 3)}
        return {"reachable": st < 500, "status": st, "url": url, "seconds": round(time.time() - t0, 3)}

    def default_probe_url(self):
        return None


def select_files(specs, select=None, cfg=None):
    """The files to fetch: select.files_regex / exclude_regex (else the default
    data and skip patterns), index files first, then by name; max_files caps."""
    select = dict(select or {})
    rx = re.compile(select.get("files_regex") or DEFAULT_DATA_REGEX)
    skip = re.compile(select.get("exclude_regex") or DEFAULT_SKIP_REGEX)
    out = [s for s in specs if rx.search(s["name"]) and not skip.search(s["name"])]
    idx = set(select.get("index_files") or [])
    out.sort(key=lambda s: (0 if s["name"] in idx or s.get("role") in ("index", "checksums") else 1, s["name"]))
    if select.get("max_files"):
        out = out[:int(select["max_files"])]
    return out


def get(cfg, name, net):
    """The provider instance of a configured provider name."""
    sec = cfg.provider(name)
    kind = sec.get("kind")
    if kind not in KINDS:
        raise ConfigError("provider %s: kind %r is not implemented (%s)" % (name, kind, KINDS))
    mod = importlib.import_module("%s.%s" % (__name__, kind))
    return mod.PROVIDER(name, sec, net, cfg)
