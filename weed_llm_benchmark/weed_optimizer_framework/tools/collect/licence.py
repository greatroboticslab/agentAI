"""The licence gate (docs/CONTINUOUS_LOOP.md §3.2 "Licence", §2.5 P6, §7.5).

Where a licence comes from, per provider (each provider module reads the field
into its candidate record, with the URL it came from as evidence):
  Zenodo metadata.license.id; Mendeley data_licence.short_name; Hugging Face
  cardData.license (else a license: tag); Kaggle licenseName (datasets/view);
  Roboflow project.license; GitHub license.spdx_id (repository level);
  the annotation index's metadata.license; the record server's license
  attribute.
When a provider's record names none, license_audit.detect_license (the
platform's slug-convention lookup for Kaggle, Roboflow, GitHub and Hugging
Face) is asked as a fallback (fallback()).

canonical(text) turns a licence text or URL into one id ("cc-by-4.0",
"cc-by-nc-sa-4.0", "cc0", "mit", ...; "unresolved" when nothing is named,
"other:<text>" for anything unrecognised). classify(id, policy) maps it with
the config's licence_policy (an id is looked up with its version, then
without it) to one of:
  permissive      usable, not research-only
  research_only   usable for research only (non-commercial and no-derivatives
                  licences): every image and every model trained on it carry
                  research_only (P6, §8)
  refused         not research-usable: the source is closed (§7.5)
  unresolved      held until a person resolves it (P6; card X16)
An "other:" id is unresolved unless the policy lists it. Nothing is
redistributed: this module and the collector never upload anywhere.
"""
from __future__ import annotations

import re

from . import LicenceError

CLASSES = ("permissive", "research_only", "refused", "unresolved")
# a restriction in the same text wins over a permissive name (inc2.splits reads licences the same way)
_RESTRICT = re.compile(r"non[\s_-]*commercial|\bnc\b|research[\s_-]*(use[\s_-]*)?only|academic[\s_-]*(use|only)|"
                       r"educational[\s_-]*(use|only|purposes)|non[\s_-]*profit|personal[\s_-]*use")
_UNRESOLVED_WORDS = ("", "unresolved", "unreachable", "unknown", "other", "none", "null", "noassertion",
                     "not specified", "see description", "other (specified in description)", "n/a", "na")


_RESTRICT_WORDS = ("noncommercial", "notforcommercial", "nocommercial", "researchonly", "researchuseonly",
                   "researchpurposesonly", "academic", "educational", "nonprofit", "personaluse", "evaluationonly")


def restricted(text):
    """True when a licence text restricts use (non-commercial in any spelling,
    research, academic, educational, non-profit, personal or evaluation use):
    the rule of inc2.splits.research_only, so the two read a licence alike."""
    t = str(text or "").lower()
    letters = re.sub(r"[^a-z]", "", t)
    return bool(re.search(r"(^|[^a-z])nc([^a-z]|$)", t) or _RESTRICT.search(t)
                or any(w in letters for w in _RESTRICT_WORDS))


def _version(t):
    m = re.search(r"(\d+\.\d+)", t)
    return m.group(1) if m else None


def canonical(text):
    """One licence id for a licence text or URL (module docstring). A
    restriction anywhere in the text wins over a permissive name in it ("CC
    BY 4.0, research use only" is research-only), as in inc2.splits."""
    lid = _canonical(text)
    if lid == "unresolved" or lid.startswith("other:"):
        return lid
    if not ("-nc" in lid or "-nd" in lid or lid in ("research-only", "cdla-sharing", "all-rights-reserved")) \
            and restricted(_text_of(text)):
        return "research-only"
    return lid


def _text_of(text):
    if isinstance(text, dict):
        return " ".join(str(text.get(k) or "") for k in ("id", "short_name", "name", "url"))
    if isinstance(text, (list, tuple)):
        return ",".join(str(x) for x in text)
    return str(text or "")


def _canonical(text):
    if text is None:
        return "unresolved"
    if isinstance(text, dict):
        text = text.get("id") or text.get("short_name") or text.get("name") or text.get("url") or ""
    if isinstance(text, (list, tuple)):
        text = ",".join(str(x) for x in text)
    t = str(text).strip().lower()
    if t in _UNRESOLVED_WORDS:
        return "unresolved"
    t = t.replace("_", "-")
    if "creativecommons.org" in t:
        m = re.search(r"creativecommons\.org/(licenses|publicdomain)/([a-z-]+)/?(\d+\.\d+)?", t)
        if m:
            if m.group(1) == "publicdomain":
                return "cc0" if "zero" in m.group(2) else "public-domain"
            kind = m.group(2).strip("-")
            return "cc-%s%s" % (kind, "-" + m.group(3) if m.group(3) else "")
    words = set(re.findall(r"[a-z0-9]+", t))
    if "cc0" in words or "cc0-1.0" in t or "public domain" in t or "publicdomain" in t or "pddl" in words:
        return "cc0" if ("cc0" in words or "cc0" in t) else "public-domain"
    if ("cc" in words and "by" in words) or "attribution" in words or "creative commons" in t or t.startswith("cc-by"):
        nc = "nc" in words or "noncommercial" in words or bool(re.search(r"non[\s_-]*commercial", t))
        sa = "sa" in words or "sharealike" in words or "share-alike" in t or "share alike" in t
        nd = "nd" in words or "noderivatives" in words or "noderivs" in words or "no derivatives" in t
        ver = _version(t)
        return "cc-by%s%s%s%s" % ("-nc" if nc else "", "-sa" if sa else "", "-nd" if nd else "",
                                  "-" + ver if ver else "")
    if "odbl" in words or "open database" in t:
        return "odbl"
    if "odc-by" in t or "odc by" in t:
        return "odc-by"
    if "cdla" in words:
        return "cdla-sharing" if "sharing" in words else "cdla-permissive"
    if _RESTRICT.search(t):
        return "research-only"                     # e.g. "MIT, for academic use only"
    if words & {"mit"}:
        return "mit"
    if "apache" in words:
        return "apache-2.0"
    if "bsd" in words:
        return "bsd-3-clause" if "3" in words else ("bsd-2-clause" if "2" in words else "bsd")
    if "agpl" in words:
        return "agpl-3.0"
    if "lgpl" in words:
        return "lgpl"
    if "gpl" in words or "gplv3" in words or "gplv2" in words:
        return "gpl-3.0" if ("3" in words or "gplv3" in words) else "gpl-2.0"
    if "unlicense" in words:
        return "unlicense"
    if "all rights reserved" in t or "proprietary" in words or "private" in words:
        return "all-rights-reserved"
    if _RESTRICT.search(t):
        return "research-only"
    return "other:%s" % re.sub(r"[^a-z0-9.+-]+", "-", t).strip("-")[:60]


def classify(lid, policy):
    """One of CLASSES for a canonical licence id under the config's policy."""
    if lid == "unresolved":
        return "unresolved"
    base = re.sub(r"-\d+(\.\d+)?$", "", lid)
    for cand in (lid, base):
        for cls in CLASSES:
            if cand in (policy.get(cls) or []):
                return cls
    return "unresolved"


def gate(text, policy, evidence=None):
    """The licence record every candidate, fetch and image carries:
    {"id", "class", "research_only", "text", "evidence"}."""
    if policy is None:
        raise LicenceError("no licence policy")
    lid = canonical(text)
    cls = classify(lid, policy)
    return {"id": lid, "class": cls, "research_only": cls == "research_only",
            "text": None if text is None else (text if isinstance(text, str) else str(text)),
            "evidence": dict(evidence or {})}


FALLBACK_KINDS = ("kaggle", "roboflow", "github", "huggingface")
DETECT = None          # tests set a stand-in for license_audit.detect_license


def fallback(slug, info=None, detect=None):
    """license_audit.detect_license(slug, info) as {"text", "evidence"} (a live
    lookup; tests inject `detect`, or set DETECT)."""
    detect = detect or DETECT
    if detect is None:
        from .. import license_audit as LA
        detect = LA.detect_license
    try:
        d = detect(slug, info or {}) or {}
    except Exception as e:  # noqa: BLE001 - detect_license promises never to raise; be safe
        return {"text": None, "evidence": {"source": "license_audit", "error": str(e)[:200]}}
    lic = d.get("license")
    return {"text": None if lic in (None, "unresolved", "unreachable") else lic,
            "evidence": {"source": "license_audit.detect_license", "license_source": d.get("license_source"),
                         "answer": lic}}


def prefer(records):
    """The index of the preferred copy among licence records of copies of one
    dataset: permissive before research_only; unresolved and refused last
    (P6: a permissive copy is preferred over a non-commercial one)."""
    rank = {"permissive": 0, "research_only": 1, "unresolved": 2, "refused": 3}
    best = None
    for i, r in enumerate(records):
        if best is None or rank.get(r.get("class"), 9) < rank.get(records[best].get("class"), 9):
            best = i
    return best


def refresh(c, policy, detect=None):
    """A candidate whose provider record named no licence, on a provider the
    platform's license_audit knows (FALLBACK_KINDS): its licence asked of
    license_audit.detect_license and gated again. Returns the candidate."""
    lic = c.get("licence") or {}
    if lic.get("class") != "unresolved" or c.get("kind", c.get("provider")) not in FALLBACK_KINDS:
        return c
    info = {"source_id": c.get("ref"), "hf_id": c.get("ref") if c.get("kind") == "huggingface" else None,
            "kaggle_ref": c.get("ref") if c.get("kind") == "kaggle" else None}
    fb = fallback(c.get("source_id"), info, detect=detect)
    if fb["text"]:
        c["licence"] = gate(fb["text"], policy, evidence=fb["evidence"])
    else:
        c["licence"] = dict(lic, fallback=fb["evidence"])
    return c

