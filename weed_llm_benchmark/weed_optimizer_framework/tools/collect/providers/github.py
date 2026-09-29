"""GitHub repositories (REST API v3; lab only: the login-node SOCKS proxy is
refused from compute nodes, §7.2).

  search    GET search/repositories?q=&per_page=
  describe  GET repos/<owner>/<repo> (license.spdx_id, size in KiB,
            default_branch, description, topics), its README (repos/../readme,
            base64) for the text, and its releases (repos/../releases)
  files     raw files a known item names (select.raw_paths: contents API ->
            download_url, checked against the git blob sha1 it gives), else
            the release assets when a release carries data (name, size,
            browser_download_url, the "sha256:<hex>" digest GitHub publishes),
            else the default branch as one archive (codeload), sized by the
            byte cap
A personal access token (config credentials, optional) lifts the rate limit;
it is sent in a header and never written. The licence is the repository's
(recorded as repository level: the data inside may differ). The source id is
the registry convention gh_<owner>__<repo>.
"""
from __future__ import annotations

import base64

from .. import ProviderError
from . import Provider, slug_part, strip_html

API = "https://api.github.com"


class GitHub(Provider):
    kind = "github"

    def source_id(self, ref):
        owner, _, name = str(ref).partition("/")
        return "gh_%s__%s" % (slug_part(owner), slug_part(name))

    def _headers(self, spec=None):
        h = {"Accept": "application/vnd.github+json"}
        tok = self.secret()
        if tok:
            h["Authorization"] = "token %s" % tok
        return h

    def _meta(self, d, sha, readme=None, releases=None):
        lic = (d.get("license") or {}).get("spdx_id")
        if lic == "NOASSERTION":
            lic = None
        m = self.meta(d.get("full_name"), version=d.get("default_branch"), title=d.get("full_name"),
                      url=d.get("html_url"),
                      description=strip_html(" ".join([str(d.get("description") or ""), readme or ""])),
                      keywords=list(d.get("topics") or []), licence_text=lic,
                      licence_evidence={"provider": self.name, "field": "license.spdx_id (repository level)",
                                        "url": "%s/repos/%s" % (self.api("base", API), d.get("full_name"))},
                      bytes=int(d.get("size") or 0) * 1024 or None,
                      gh_default_branch=d.get("default_branch"), gh_releases=releases or [])
        m["raw_sha256"].append(sha)
        return m

    def search(self, query, limit):
        st, body, sha, _h = self.net.get_json("%s/search/repositories" % self.api("base", API),
                                              {"q": query, "per_page": str(limit)}, headers=self._headers(),
                                              what="github search")
        if st != 200 or not isinstance(body, dict):
            raise ProviderError("github search %r answered %s" % (query, st))
        return [self._meta(d, sha) for d in body.get("items") or [] if d.get("full_name")]

    def describe(self, ref):
        base = self.api("base", API)
        st, body, sha, _h = self.net.get_json("%s/repos/%s" % (base, ref), headers=self._headers(),
                                              what="github %s" % ref)
        if st != 200 or not isinstance(body, dict):
            raise ProviderError("github repository %s answered %s" % (ref, st))
        readme = None
        st2, rd, _s2, _h2 = self.net.get_json("%s/repos/%s/readme" % (base, ref), headers=self._headers(),
                                              what="github readme %s" % ref)
        if st2 == 200 and isinstance(rd, dict) and rd.get("content"):
            try:
                readme = base64.b64decode(rd["content"]).decode("utf-8", "replace")[:8000]
            except ValueError:
                readme = None
        rels = []
        st3, rl, _s3, _h3 = self.net.get_json("%s/repos/%s/releases" % (base, ref), {"per_page": "10"},
                                              headers=self._headers(), what="github releases %s" % ref)
        if st3 == 200 and isinstance(rl, list):
            for r in rl:
                for a in r.get("assets") or []:
                    dig = str(a.get("digest") or "")
                    rels.append({"tag": r.get("tag_name"), "name": a.get("name"), "size": a.get("size"),
                                 "url": a.get("browser_download_url"),
                                 "sha256": dig.split(":", 1)[1] if dig.startswith("sha256:") else None})
        return self._meta(body, sha, readme=readme, releases=rels)

    def files(self, meta, select=None):
        raw = list((select or {}).get("raw_paths") or [])
        if raw:
            branch = meta.get("gh_default_branch") or meta.get("version") or "main"
            out = []
            for path in raw:
                st, d, _sha, _h = self.net.get_json("%s/repos/%s/contents/%s" % (self.api("base", API), meta["ref"],
                                                                                 path), {"ref": branch},
                                                    headers=self._headers(), what="github contents %s" % path)
                if st != 200 or not isinstance(d, dict) or not d.get("download_url"):
                    raise ProviderError("github %s: no raw file %s (%s)" % (meta["ref"], path, st))
                out.append({"name": path, "url": d["download_url"], "size": d.get("size"),
                            "checksums": {"git-blob-sha1": d["sha"]} if d.get("sha") else {}, "role": "data"})
            return out
        rels = [r for r in meta.get("gh_releases") or [] if r.get("url")]
        if rels:
            return [{"name": "%s/%s" % (r["tag"], r["name"]), "url": r["url"], "size": r.get("size"),
                     "checksums": {"sha256": r["sha256"]} if r.get("sha256") else {}, "role": "data"}
                    for r in rels]
        branch = meta.get("gh_default_branch") or meta.get("version") or "main"
        return [{"name": "%s-%s.zip" % (meta["ref"].split("/")[-1], branch),
                 "url": "https://codeload.github.com/%s/zip/refs/heads/%s" % (meta["ref"], branch),
                 "size": None, "checksums": {}, "role": "data", "whole_dataset": True}]

    def default_probe_url(self):
        return "%s/rate_limit" % self.api("base", API)


PROVIDER = GitHub
