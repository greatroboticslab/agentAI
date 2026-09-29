"""A record server with an export API and an FTP server for the data (the hosts
and the FTP login come from the config; checked 2026-09-28).

  search    none (the export API has no search the collector can use): the
            known items name the records, and discovery reports the miss
  describe  GET <base>/services/export/node/<id>?format=json&attrspec=all :
            title, description, keywords and the licence attribute
  files     an FTP listing, driven by the known item's select block:
              root         the directory whose subdirectories are class codes
              codes        "targets": keep the subdirectories whose name is an
                           EPPO code of a target (EPPO table -> binomial ->
                           target taxon; a genus-rank target keeps every code
                           of its genus), or an explicit list of codes
              index_files  files fetched first (e.g. the box table)
              checksums_file
                           a "<hex>  ./<path>" list; its algorithm from the
                           hex length (sha512, sha256, sha1, md5)
            every selected file carries its size (SIZE) and published checksum
Downloads go through transport.FtpSession (capped, checked, atomic).
"""
from __future__ import annotations

import re

from .. import ProviderError
from . import Provider, strip_html

_ALGO_BY_LEN = {128: "sha512", 64: "sha256", 40: "sha1", 32: "md5"}


def parse_checksums(text):
    """{path without a leading ./: (algo, hex)} of a checksum list."""
    out = {}
    for ln in text.splitlines():
        m = re.match(r"^([0-9a-fA-F]{32,128})\s+\*?(.+?)\s*$", ln)
        if not m:
            continue
        hx = m.group(1).lower()
        algo = _ALGO_BY_LEN.get(len(hx))
        if algo:
            out[re.sub(r"^\./", "", m.group(2))] = (algo, hx)
    return out


class RecordFtp(Provider):
    kind = "mediatum"
    searchable = False

    def base(self):
        return str(self.sec["base_url"]).rstrip("/")

    def source_id(self, ref):
        return "%s_%s" % (self.name, ref)

    def describe(self, ref):
        st, d, sha, _h = self.net.get_json("%s/services/export/node/%s" % (self.base(), ref),
                                           {"format": "json", "attrspec": "all"}, what="%s %s" % (self.name, ref))
        if st != 200 or not isinstance(d, dict) or not d.get("nodelist"):
            raise ProviderError("%s record %s answered %s" % (self.name, ref, st))
        node = d["nodelist"][0][0] if isinstance(d["nodelist"][0], list) else d["nodelist"][0]
        a = node.get("attributes") or {}
        kws = [k.strip() for k in re.split(r"[;,]", str(a.get("keywords") or "")) if k.strip()]
        m = self.meta(ref, title=a.get("title"), url="%s/%s" % (self.base(), ref),
                      description=strip_html(a.get("description")), keywords=kws,
                      licence_text=a.get("license") or a.get("license_other"),
                      licence_evidence={"provider": self.name, "field": "attributes.license",
                                        "url": "%s/services/export/node/%s" % (self.base(), ref)})
        m["raw_sha256"].append(sha)
        return m

    def _login(self, select):
        acc = dict(self.sec.get("ftp") or {})
        acc.update((select or {}).get("ftp") or {})
        if not acc.get("host"):
            raise ProviderError("%s: no FTP host in the config" % self.name)
        return self.net.ftp(acc["host"], acc.get("user") or "anonymous", acc.get("password") or "")

    def files(self, meta, select=None):
        select = dict(select or {})
        root = select.get("root")
        if not root:
            raise ProviderError("%s %s: the known item's select block names no root directory"
                                % (self.name, meta["ref"]))
        ses = self._login(select)
        try:
            sums = {}
            cf = select.get("checksums_file")
            if cf:
                sums = parse_checksums(ses.read(cf, max_bytes=64 << 20).decode("utf-8", "replace"))
            codes = select.get("codes")
            is_target = select.get("_is_target_code")
            dirs = []
            for d in ses.nlst(root):
                code = d.rstrip("/").split("/")[-1]
                if isinstance(codes, list):
                    keep = code in codes
                elif codes == "targets" and is_target is not None:
                    keep = bool(is_target(code))
                else:
                    keep = False
                if keep:
                    dirs.append(d.rstrip("/"))
            out = []
            for f in select.get("index_files") or []:
                algo_hex = sums.get(f)
                out.append({"name": f, "ftp_path": f, "size": ses.size(f),
                            "checksums": {algo_hex[0]: algo_hex[1]} if algo_hex else {}, "role": "index"})
            for d in dirs:
                for p in ses.nlst(d):
                    rel = p.lstrip("./")
                    algo_hex = sums.get(rel)
                    out.append({"name": rel, "ftp_path": p, "size": ses.size(p),
                                "checksums": {algo_hex[0]: algo_hex[1]} if algo_hex else {}, "role": "data",
                                "group": d.split("/")[-1]})
            return out
        finally:
            ses.close()

    def fetch_file(self, spec, dest, max_bytes=None, select=None):
        ses = self._login(select)
        try:
            return ses.download(spec["ftp_path"], dest, max_bytes=max_bytes, expected_size=spec.get("size"),
                                checksums=spec.get("checksums"))
        finally:
            ses.close()

    def default_probe_url(self):
        return "%s/services/export/node/1?format=json" % self.base()


PROVIDER = RecordFtp
