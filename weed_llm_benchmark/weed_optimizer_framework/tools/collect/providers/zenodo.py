"""Zenodo (records API; checked 2026-09-28): search GET <records>?q=&size=&type=dataset,
a record GET <records>/<id>; files carry key, size, "md5:<hex>" and links.self.
The endpoint is funnel.fetch's (ZENODO_RECORD), used as a library. The licence
is metadata.license.id. The creators and their affiliations join the
keywords, so a lab group rule can recognise the authors. Zenodo declares no
class list: the text decides.
"""
from __future__ import annotations

from .. import ProviderError
from . import Provider, strip_html


def _records_url():
    from ...funnel import fetch as FF
    return FF.ZENODO_RECORD.rsplit("/", 1)[0]          # https://zenodo.org/api/records


class Zenodo(Provider):
    kind = "zenodo"

    def source_id(self, ref):
        return "zenodo_%s" % str(ref)

    def _meta(self, rec, sha=None, query=None):
        md = rec.get("metadata") or {}
        files = rec.get("files") or []
        lic = (md.get("license") or {}).get("id") if isinstance(md.get("license"), dict) else md.get("license")
        rt = (md.get("resource_type") or {}).get("type")
        people = []
        for cr in md.get("creators") or []:
            if isinstance(cr, dict):
                people += [x for x in (cr.get("name"), cr.get("affiliation")) if x]
        m = self.meta(rec.get("id") or rec.get("recid"), version=str(rec.get("revision") or "") or None,
                      title=md.get("title") or rec.get("title"),
                      url=(rec.get("links") or {}).get("self_html") or rec.get("doi_url"),
                      description=strip_html(md.get("description")),
                      keywords=list(md.get("keywords") or []) + people,
                      licence_text=lic, licence_evidence={"provider": self.name, "field": "metadata.license.id",
                                                          "url": "%s/%s" % (_records_url(), rec.get("id"))},
                      bytes=sum(int(f.get("size") or 0) for f in files) or None,
                      resource_type=rt, zenodo_files=[{"key": f.get("key"), "size": f.get("size"),
                                                       "checksum": f.get("checksum"),
                                                       "url": (f.get("links") or {}).get("self")} for f in files])
        if sha:
            m["raw_sha256"].append(sha)
        if rt and rt != "dataset":
            m["annotation_evidence"].append("resource_type:%s" % rt)
        return m

    def search(self, query, limit):
        st, body, sha, _h = self.net.get_json(_records_url(), {"q": query, "size": str(limit), "type": "dataset",
                                                               "sort": "bestmatch"}, what="zenodo search")
        if st != 200 or not isinstance(body, dict):
            raise ProviderError("zenodo search %r answered %s" % (query, st))
        return [self._meta(r, sha, query) for r in (body.get("hits") or {}).get("hits") or []]

    def describe(self, ref):
        st, body, sha, _h = self.net.get_json("%s/%s" % (_records_url(), ref), what="zenodo record %s" % ref)
        if st != 200 or not isinstance(body, dict):
            raise ProviderError("zenodo record %s answered %s" % (ref, st))
        return self._meta(body, sha)

    def files(self, meta, select=None):
        out = []
        for f in meta.get("zenodo_files") or []:
            ck = str(f.get("checksum") or "")
            algo, _, val = ck.partition(":")
            out.append({"name": f["key"], "url": f["url"], "size": f.get("size"),
                        "checksums": {algo: val} if algo in ("md5", "sha256", "sha1") and val else {},
                        "role": "data"})
        return out

    def default_probe_url(self):
        return "%s?size=1" % _records_url()


PROVIDER = Zenodo
