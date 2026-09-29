"""Roboflow Universe (REST API with the platform's key; the search endpoint and
its fields as recorded in roboflow_source.py on 2026-06-08).

  search    GET universe/search?q=&api_key=&page= : name, url (.../<ws>/<proj>),
            workspace, type, classes[], classCount, images, latestVersion,
            license. The only provider that declares class lists before any
            download, so class-name search starts here (§7.3).
  describe  GET <ws>/<proj>?api_key= : project.classes {name: box count},
            project.license, project.type, project.images, versions
  files     one file: the export of the latest version in the configured
            format (GET <ws>/<proj>/<version>/<format>?api_key= answers
            export.link; an export still being generated is polled a bounded
            number of times). Roboflow publishes no checksum.
The key comes from the config's credentials (environment or key file), is
sent as a parameter, and never reaches a record (transport.redact). Nothing
is ever uploaded: the collector reads Roboflow and never writes to it.
"""
from __future__ import annotations

import time

from .. import CredentialsMissing, ProviderError
from . import Provider, slug_part, strip_html

API = "https://api.roboflow.com"


class Roboflow(Provider):
    kind = "roboflow"

    def source_id(self, ref):
        ws, _, proj = str(ref).partition("/")
        return "rf_%s__%s" % (slug_part(ws).replace("_", "-"), slug_part(proj).replace("_", "-"))

    def _key(self):
        k = self.secret()
        if not k:
            raise CredentialsMissing(self.credentials()[1])
        return k

    def _meta_search(self, r, sha):
        url = str(r.get("url") or "").rstrip("/")
        ws = ((r.get("workspace") or {}).get("url")) or (url.split("/")[-2] if url.count("/") >= 1 else "")
        proj = url.split("/")[-1] if url else ""
        if not ws or not proj:
            return None
        classes = [{"id": i, "name": str(c), "hints": []} for i, c in enumerate(r.get("classes") or [])]
        typ = r.get("type")
        m = self.meta("%s/%s" % (ws, proj), version=str(r.get("latestVersion") or "") or None, title=r.get("name"),
                      url=url or None, description=strip_html(r.get("description")),
                      licence_text=r.get("license"),
                      licence_evidence={"provider": self.name, "field": "license (universe search)"},
                      classes=classes or None, classes_source="declared" if classes else "none",
                      annotation="boxes" if typ == "object-detection" else (
                          "image_level" if typ == "classification" else "unknown"),
                      annotation_evidence=["type:%s" % typ] if typ else [], images=r.get("images"))
        m["raw_sha256"].append(sha)
        return m

    def search(self, query, limit):
        out = []
        for page in range(1, 11):
            st, body, sha, _h = self.net.get_json("%s/universe/search" % self.api("base", API),
                                                  {"q": query, "api_key": self._key(), "page": str(page)},
                                                  what="roboflow search")
            if st != 200 or not isinstance(body, dict):
                if page == 1:
                    raise ProviderError("roboflow search %r answered %s" % (query, st))
                break
            batch = body.get("results") or []
            for r in batch:
                m = self._meta_search(r, sha)
                if m is not None:
                    out.append(m)
            if len(out) >= limit or len(batch) < int(body.get("page_size", 12) or 12):
                break
        return out[:limit]

    def describe(self, ref):
        st, body, sha, _h = self.net.get_json("%s/%s" % (self.api("base", API), ref), {"api_key": self._key()},
                                              what="roboflow %s" % ref)
        if st != 200 or not isinstance(body, dict):
            raise ProviderError("roboflow project %s answered %s" % (ref, st))
        p = body.get("project") or {}
        cls = p.get("classes") or {}
        classes = [{"id": i, "name": str(n), "hints": [], "boxes": int(c) if isinstance(c, (int, float)) else None}
                   for i, (n, c) in enumerate(sorted(cls.items()))] if isinstance(cls, dict) else \
            [{"id": i, "name": str(n), "hints": []} for i, n in enumerate(cls)]
        vers = body.get("versions") or []
        ver = str(vers[-1].get("id", "")).split("/")[-1] if vers and isinstance(vers[-1], dict) else None
        typ = p.get("type")
        m = self.meta(ref, version=ver, title=p.get("name") or ref, url="https://universe.roboflow.com/%s" % ref,
                      description=strip_html(p.get("description") or p.get("annotation")),
                      licence_text=p.get("license") or body.get("license"),
                      licence_evidence={"provider": self.name, "field": "project.license",
                                        "url": "%s/%s" % (self.api("base", API), ref)},
                      classes=classes or None, classes_source="declared" if classes else "none",
                      annotation="boxes" if typ == "object-detection" else (
                          "image_level" if typ == "classification" else "unknown"),
                      annotation_evidence=["type:%s" % typ] if typ else [], images=p.get("images"))
        m["raw_sha256"].append(sha)
        return m

    def files(self, meta, select=None):
        ver = (select or {}).get("version") or meta.get("version")
        if not ver:
            raise ProviderError("roboflow %s has no generated version to export" % meta["ref"])
        fmt = self.sec.get("export_format") or "yolov8"
        url = "%s/%s/%s/%s" % (self.api("base", API), meta["ref"], ver, fmt)
        link = None
        for _i in range(int(self.sec.get("export_polls") or 6)):
            st, body, _sha, _h = self.net.get_json(url, {"api_key": self._key()}, what="roboflow export")
            if st == 200 and isinstance(body, dict):
                link = (body.get("export") or {}).get("link")
                if link:
                    break
            time.sleep(float(self.sec.get("export_poll_s") or 10))
        if not link:
            raise ProviderError("roboflow %s v%s: no export link in the %s format" % (meta["ref"], ver, fmt))
        return [{"name": "%s-v%s-%s.zip" % (meta["ref"].split("/")[-1], ver, fmt), "url": link, "size": None,
                 "checksums": {}, "role": "data", "whole_dataset": True}]

    def default_probe_url(self):
        return "%s/" % self.api("base", API)


PROVIDER = Roboflow
