"""Hugging Face datasets (Hub API; checked 2026-09-28).

  search    GET api/datasets?search=&limit=&full=true
  describe  GET api/datasets/<repo>?blobs=true (funnel.fetch's HF_API): the card
            data (license, task_categories, dataset_info), tags, siblings with
            size, blobId and, for LFS files, lfs.sha256
  files     the siblings; resolve/<sha>/<path> downloads, checked against the
            LFS sha256, else against the git blob sha1
Declared classes are the ClassLabel names of the card's dataset_info features
(often numeric: then a card class table or nothing makes them targets). The
annotation kind comes from task_categories/tags: object-detection is boxes,
image-classification alone is image-level. The source id is the registry's
convention for this provider, <owner>__<name> in lower case.
"""
from __future__ import annotations

from .. import ProviderError
from . import Provider, slug_part, strip_html

SEARCH_URL = "https://huggingface.co/api/datasets"
BOX_TAGS = ("task_categories:object-detection", "task_ids:object-detection", "object-detection")
IMG_TAGS = ("task_categories:image-classification", "image-classification")


def _ff():
    from ...funnel import fetch as FF
    return FF


def class_labels(card):
    """[(id, name)] of the first ClassLabel found in dataset_info.features."""
    info = (card or {}).get("dataset_info")
    infos = info if isinstance(info, list) else [info]
    for inf in infos:
        found = _find_labels((inf or {}).get("features"))
        if found:
            return found
    return []


def _find_labels(node):
    if isinstance(node, dict):
        cl = node.get("class_label")
        if isinstance(cl, dict) and isinstance(cl.get("names"), (dict, list)):
            names = cl["names"]
            if isinstance(names, dict):
                return sorted(((int(k), str(v)) for k, v in names.items()), key=lambda x: x[0])
            return list(enumerate(str(v) for v in names))
        for v in node.values():
            r = _find_labels(v)
            if r:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _find_labels(v)
            if r:
                return r
    return []


class HuggingFace(Provider):
    kind = "huggingface"

    def source_id(self, ref):
        owner, _, name = str(ref).partition("/")
        return "%s__%s" % (slug_part(owner), slug_part(name))

    def _headers(self, spec=None):
        tok = self.secret()
        return {"Authorization": "Bearer %s" % tok} if tok else None

    def _meta(self, d, sha):
        card = d.get("cardData") or {}
        tags = [str(t) for t in d.get("tags") or []]
        lic = card.get("license")
        if isinstance(lic, list):
            lic = ",".join(str(x) for x in lic)
        if not lic:
            lic = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)
        labels = class_labels(card)
        ann, ev = "unknown", []
        if any(t in tags for t in BOX_TAGS) or "object-detection" in (card.get("task_categories") or []):
            ann, ev = "boxes", ["tag:object-detection"]
        elif any(t in tags for t in IMG_TAGS):
            ann, ev = "image_level", ["tag:image-classification"]
        sib = d.get("siblings") or []
        size = sum(int(s.get("size") or 0) for s in sib) or d.get("usedStorage")
        n = None
        for inf in (card.get("dataset_info") if isinstance(card.get("dataset_info"), list) else [card.get("dataset_info")]):
            for sp in (inf or {}).get("splits") or []:
                n = (n or 0) + int(sp.get("num_examples") or 0)
        m = self.meta(d.get("id"), version=d.get("sha"), title=d.get("id"),
                      url="https://huggingface.co/datasets/%s" % d.get("id"),
                      description=strip_html(d.get("description")), keywords=tags,
                      licence_text=lic, licence_evidence={"provider": self.name, "field": "cardData.license",
                                                          "url": _ff().HF_API % d.get("id")},
                      classes=[{"id": i, "name": nm, "hints": []} for i, nm in labels] or None,
                      classes_source="declared" if labels else "none", annotation=ann, annotation_evidence=ev,
                      bytes=size or None, images=n,
                      hf_siblings=[{"path": s.get("rfilename"), "size": s.get("size"), "blob": s.get("blobId"),
                                    "lfs": (s.get("lfs") or {}).get("sha256")} for s in sib])
        m["raw_sha256"].append(sha)
        return m

    def search(self, query, limit):
        st, body, sha, _h = self.net.get_json(self.api("search", SEARCH_URL),
                                              {"search": query, "limit": str(limit), "full": "true"},
                                              headers=self._headers(), what="huggingface search")
        if st != 200 or not isinstance(body, list):
            raise ProviderError("huggingface search %r answered %s" % (query, st))
        return [self._meta(d, sha) for d in body if isinstance(d, dict) and d.get("id")]

    def describe(self, ref):
        st, body, sha, _h = self.net.get_json(_ff().HF_API % ref, {"blobs": "true"}, headers=self._headers(),
                                              what="huggingface %s" % ref)
        if st != 200 or not isinstance(body, dict):
            raise ProviderError("huggingface dataset %s answered %s" % (ref, st))
        return self._meta(body, sha)

    def files(self, meta, select=None):
        rev = meta.get("version") or "main"
        out = []
        for s in meta.get("hf_siblings") or []:
            if not s.get("path"):
                continue
            ck = {"sha256": s["lfs"]} if s.get("lfs") else ({"git-blob-sha1": s["blob"]} if s.get("blob") else {})
            out.append({"name": s["path"], "url": "https://huggingface.co/datasets/%s/resolve/%s/%s"
                        % (meta["ref"], rev, s["path"]), "size": s.get("size"), "checksums": ck, "role": "data"})
        return out

    def default_probe_url(self):
        return "%s?limit=1" % self.api("search", SEARCH_URL)


PROVIDER = HuggingFace
