"""An annotation index whose documents are single images with their category
names (the host comes from the config's base_url; checked 2026-09-28).

  search    GET <base>/api/set_csrf/ (a CSRF cookie), then POST
            <base>/elasticsearch/<index>/_msearch (the only query the site
            forwards) with a match_phrase on the category names and a terms
            aggregation on the upload id: one candidate per upload, with the
            number of matching images
  describe  GET <base>/api/upload_info/<upload id>: metadata (name, licence URL,
            description, creators and their affiliations) and agcontexts, whose
            category_statistics give every category's image and box counts:
            declared classes with box counts before any download
  files     one file: <base>/code/download/<upload id>.zip (its size from a
            HEAD request when the server gives one; no checksum is published)
Category names have the form "<role>: <taxon> (<qualifier>)"; the taxon is
passed as a hint to the class map. The config's taxon template drives the
search (the binomial, lower case). The source id is <provider>_<upload id>.
"""
from __future__ import annotations

import json

from .. import ProviderError
from ..classmap import role_taxon_hint
from . import Provider, strip_html


class AnnotationIndex(Provider):
    kind = "weedai"

    def base(self):
        return str(self.sec["base_url"]).rstrip("/")

    def source_id(self, ref):
        return "%s_%s" % (self.name, str(ref).lower())

    def _csrf(self):
        st, _raw, _h = self.net.request("%s/api/set_csrf/" % self.base())
        tok = self.net.cookie("csrftoken")
        if st != 200 or not tok:
            raise ProviderError("%s: no CSRF cookie (status %s)" % (self.name, st))
        return tok

    def search(self, query, limit):
        tok = self._csrf()
        index = self.sec.get("index") or "weedid"
        body = "{}\n" + json.dumps({"query": {"match_phrase": {"annotations.category.name": query.lower()}},
                                    "size": 0, "aggs": {"u": {"terms": {"field": "upload_id.keyword",
                                                                          "size": int(limit)}}}}) + "\n"
        st, res, sha, _h = self.net.get_json(
            "%s/elasticsearch/%s/_msearch" % (self.base(), index), data=body.encode("utf-8"), method="POST",
            headers={"X-CSRFToken": tok, "Referer": "%s/explore" % self.base(),
                     "Content-Type": "application/x-ndjson"}, what="%s search" % self.name)
        if st != 200 or not isinstance(res, dict):
            raise ProviderError("%s search %r answered %s" % (self.name, query, st))
        r0 = (res.get("responses") or [{}])[0]
        if r0.get("error"):
            raise ProviderError("%s search %r: %s" % (self.name, query, str(r0["error"])[:300]))
        out = []
        for b in ((r0.get("aggregations") or {}).get("u") or {}).get("buckets") or []:
            m = self.meta(b.get("key"), matched_images=b.get("doc_count"))
            m["raw_sha256"].append(sha)
            out.append(m)
        return out

    def describe(self, ref):
        st, d, sha, _h = self.net.get_json("%s/api/upload_info/%s" % (self.base(), ref),
                                           what="%s %s" % (self.name, ref))
        if st != 200 or not isinstance(d, dict):
            raise ProviderError("%s upload %s answered %s" % (self.name, ref, st))
        md = d.get("metadata") or {}
        stats = {}
        images = 0
        for a in d.get("agcontexts") or []:
            images += int(a.get("n_images") or 0)
            for name, s in (a.get("category_statistics") or {}).items():
                cur = stats.setdefault(name, {"boxes": 0, "images": 0})
                cur["boxes"] += int((s or {}).get("bounding_box_count") or 0)
                cur["images"] += int((s or {}).get("image_count") or 0)
        classes = []
        for i, name in enumerate(sorted(stats)):
            h = role_taxon_hint(name)
            classes.append({"id": i, "name": name, "hints": [h] if h else [], "boxes": stats[name]["boxes"],
                            "images": stats[name]["images"]})
        boxes = sum(v["boxes"] for v in stats.values())
        affs = sorted({((c or {}).get("affiliation") or {}).get("name") for c in md.get("creator") or []} - {None})
        m = self.meta(ref, version=str(d.get("head_version") or "") or None, title=md.get("name"),
                      url="%s/datasets/%s" % (self.base(), ref), description=strip_html(md.get("description")),
                      keywords=affs, licence_text=md.get("license"),
                      licence_evidence={"provider": self.name, "field": "metadata.license",
                                        "url": "%s/api/upload_info/%s" % (self.base(), ref)},
                      classes=classes or None, classes_source="declared" if classes else "none",
                      annotation="boxes" if boxes else "unknown",
                      annotation_evidence=["bounding_box_count:%d" % boxes] if boxes else [],
                      images=images or None, affiliations=affs)
        m["raw_sha256"].append(sha)
        st2, _raw, hd = self.net.request("%s/code/download/%s.zip" % (self.base(), ref), method="HEAD")
        cl = {str(k).lower(): v for k, v in (hd or {}).items()}.get("content-length")
        if st2 == 200 and cl and str(cl).isdigit():
            m["bytes"] = int(cl)
        return m

    def files(self, meta, select=None):
        return [{"name": "%s.zip" % meta["ref"], "url": "%s/code/download/%s.zip" % (self.base(), meta["ref"]),
                 "size": meta.get("bytes"), "checksums": {}, "role": "data", "whole_dataset": True}]

    def default_probe_url(self):
        return "%s/api/set_csrf/" % self.base()


PROVIDER = AnnotationIndex
