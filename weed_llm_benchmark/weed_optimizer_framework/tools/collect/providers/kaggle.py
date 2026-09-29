"""Kaggle datasets (API v1 with a bearer token, or basic auth from kaggle.json).

  search    GET api/v1/datasets/list?search=&page=1
  describe  GET api/v1/datasets/view/<owner>/<name>: title, subtitle,
            description, keywords/tags, licenseName, totalBytes
  files     one file: the dataset archive, GET api/v1/datasets/download/<owner>/<name>
            (a redirect to storage). Kaggle publishes no checksum: the sha256
            of what arrived is recorded, and the byte cap bounds the stream.
Credentials (config providers.<name>.credentials): a bearer token from the
environment or a token file, else basic auth from kaggle.json ({"username",
"key"}). Missing both: the source is held (card X16). The token is sent in a
header and never written. The source id is the registry convention
kg_<owner>__<name>. Kaggle declares no class list: the text decides.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

from .. import CredentialsMissing, ProviderError
from . import Provider, slug_part, strip_html

API = "https://www.kaggle.com/api/v1"


class Kaggle(Provider):
    kind = "kaggle"

    def source_id(self, ref):
        owner, _, name = str(ref).partition("/")
        return "kg_%s__%s" % (slug_part(owner), slug_part(name))

    def _auth(self):
        cr = self.sec.get("credentials") or {}
        from ..transport import read_secret
        tok, _w = read_secret(cr.get("env") or (), cr.get("files") or ())
        if tok:
            return {"Authorization": "Bearer %s" % tok}
        for f in cr.get("basic_files") or ():
            p = Path(os.path.expanduser(str(f)))
            try:
                d = json.loads(p.read_text())
                if d.get("username") and d.get("key"):
                    raw = ("%s:%s" % (d["username"], d["key"])).encode("utf-8")
                    return {"Authorization": "Basic %s" % base64.b64encode(raw).decode("ascii")}
            except (OSError, ValueError):
                continue
        return None

    def credentials(self):
        if self._auth():
            return True, "present"
        cr = self.sec.get("credentials") or {}
        return False, "provider %s needs a token (%s) or a kaggle.json (%s) (card X16)" % (
            self.name, ", ".join(["$%s" % e for e in cr.get("env") or ()] + list(cr.get("files") or ())),
            ", ".join(cr.get("basic_files") or ()))

    def _headers(self, spec=None):
        h = self._auth()
        if h is None:
            raise CredentialsMissing(self.credentials()[1])
        return h

    def _meta(self, d, sha):
        ref = d.get("ref") or "%s/%s" % (d.get("ownerRef") or d.get("ownerName"), d.get("datasetSlug"))
        tags = [t.get("name") if isinstance(t, dict) else str(t) for t in (d.get("tags") or [])]
        kws = list(d.get("keywords") or []) + [t for t in tags if t]
        lic = d.get("licenseName") or ((d.get("licenses") or [{}])[0] or {}).get("name")
        m = self.meta(ref, version=str(d.get("currentVersionNumber") or "") or None, title=d.get("title"),
                      url="https://www.kaggle.com/datasets/%s" % ref,
                      description=strip_html(" ".join(str(d.get(k) or "") for k in ("subtitle", "description"))),
                      keywords=kws, licence_text=lic,
                      licence_evidence={"provider": self.name, "field": "licenseName",
                                        "url": "%s/datasets/view/%s" % (self.api("base", API), ref)},
                      bytes=d.get("totalBytes"))
        m["raw_sha256"].append(sha)
        return m

    def search(self, query, limit):
        st, body, sha, _h = self.net.get_json("%s/datasets/list" % self.api("base", API),
                                              {"search": query, "page": "1"}, headers=self._headers(),
                                              what="kaggle search")
        if st != 200 or not isinstance(body, list):
            raise ProviderError("kaggle search %r answered %s" % (query, st))
        return [self._meta(d, sha) for d in body[:limit] if isinstance(d, dict)]

    def describe(self, ref):
        st, body, sha, _h = self.net.get_json("%s/datasets/view/%s" % (self.api("base", API), ref),
                                              headers=self._headers(), what="kaggle %s" % ref)
        if st != 200 or not isinstance(body, dict):
            raise ProviderError("kaggle dataset %s answered %s" % (ref, st))
        return self._meta(body, sha)

    def files(self, meta, select=None):
        return [{"name": "%s.zip" % meta["ref"].split("/")[-1],
                 "url": "%s/datasets/download/%s" % (self.api("base", API), meta["ref"]),
                 "size": None, "checksums": {}, "role": "data", "whole_dataset": True}]

    def default_probe_url(self):
        return "%s/datasets/list?search=probe&page=1" % self.api("base", API)


PROVIDER = Kaggle
