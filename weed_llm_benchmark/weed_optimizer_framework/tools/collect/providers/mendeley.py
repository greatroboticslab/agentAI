"""Mendeley Data (checked 2026-09-28).

  search    GET <search>?query=&type=DATASET&size= (the site's research-data
            search; records outside Mendeley Data are dropped: only a
            data.mendeley.com/datasets/<id> record has the public API below)
  describe  GET public-api/datasets/<id>[?version=v] (funnel.fetch's
            MENDELEY_DATASET): name, description, data_licence.short_name,
            version
  files     funnel.fetch._mendeley_listing (folders, then files per folder).
            The files listing answers 206 with "Content-Range: items a-b/total"
            for a folder of more than 1,000 files and cannot be paged: such a
            folder is partially listed, and a selection that could reach into
            it is refused (as funnel.fetch refuses it).
Downloads are checked against the listing's sha256.
"""
from __future__ import annotations

import re

from .. import ProviderError
from . import Provider, strip_html

SEARCH_URL = "https://data.mendeley.com/api/research-data/search"
_ID = re.compile(r"data\.mendeley\.com/datasets/([a-z0-9]+)(?:/(\d+))?")


def _ff():
    from ...funnel import fetch as FF
    return FF


class Mendeley(Provider):
    kind = "mendeley"

    def source_id(self, ref):
        ds, _, ver = str(ref).partition("/")
        return "mendeley_%s%s" % (ds, "_v%s" % ver if ver else "")

    def _transport(self):
        return lambda url, params: self.net.request(url, params)

    def search(self, query, limit):
        st, body, sha, _h = self.net.get_json(self.api("search", SEARCH_URL),
                                              {"query": query, "type": "DATASET", "size": str(limit)},
                                              what="mendeley search")
        if st != 200 or not isinstance(body, dict):
            raise ProviderError("mendeley search %r answered %s" % (query, st))
        out = []
        for r in body.get("records") or []:
            m = _ID.search(str(r.get("url") or ""))
            if not m:
                continue
            meta = self.meta(m.group(1), title=r.get("title"), url=r.get("url"),
                             description=strip_html(r.get("description")))
            meta["raw_sha256"].append(sha)
            out.append(meta)
        return out

    def describe(self, ref):
        FF = _ff()
        ds, _, ver = str(ref).partition("/")
        params = {"version": ver} if ver else {}
        st, body, sha, _h = self.net.get_json(FF.MENDELEY_DATASET % ds, params, what="mendeley %s" % ds)
        if st != 200 or not isinstance(body, dict):
            raise ProviderError("mendeley dataset %s answered %s" % (ref, st))
        ver = str(body.get("version") or ver or "")
        cats = [c.get("label") for c in body.get("categories") or [] if isinstance(c, dict)]
        cats += [i.get("name") for i in body.get("institutions") or [] if isinstance(i, dict) and i.get("name")]
        m = self.meta("%s/%s" % (ds, ver) if ver else ds, version=ver or None, title=body.get("name"),
                      url="https://data.mendeley.com/datasets/%s/%s" % (ds, ver),
                      description=strip_html(body.get("description")), keywords=cats,
                      licence_text=(body.get("data_licence") or {}).get("short_name"),
                      licence_evidence={"provider": self.name, "field": "data_licence.short_name",
                                        "url": FF.MENDELEY_DATASET % ds},
                      bytes=body.get("size") if isinstance(body.get("size"), (int, float)) else None)
        m["source_id"] = self.source_id("%s/%s" % (ds, ver) if ver else ds)
        m["raw_sha256"].append(sha)
        return m

    def files(self, meta, select=None):
        FF = _ff()
        ds, _, ver = str(meta["ref"]).partition("/")
        if not ver:
            raise ProviderError("mendeley %s: a version is needed to list files" % ds)
        listing, _raw, partial = FF._mendeley_listing(self._transport(), ds, ver)
        frx = re.compile((select or {}).get("folder_regex")) if (select or {}).get("folder_regex") else None
        in_scope = sorted(f for f in partial if frx is None or frx.search(f))
        if in_scope:
            raise ProviderError("mendeley %s v%s lists only part of folder(s) %s (206, not pageable); narrow the "
                                "selection with folder_regex" % (ds, ver, in_scope))
        out = []
        for folder, files in sorted(listing.items()):
            if frx is not None and not frx.search(folder):
                continue
            for f in files:
                cd = f.get("content_details") or {}
                if not cd.get("download_url"):
                    continue
                name = "%s/%s" % (folder, f.get("filename")) if folder else f.get("filename")
                out.append({"name": name, "url": cd["download_url"], "size": f.get("size") or cd.get("size"),
                            "checksums": {"sha256": cd["sha256_hash"]} if cd.get("sha256_hash") else {},
                            "role": "data"})
        return out

    def default_probe_url(self):
        return "%s?query=probe&type=DATASET&size=1" % self.api("search", SEARCH_URL)


PROVIDER = Mendeley
