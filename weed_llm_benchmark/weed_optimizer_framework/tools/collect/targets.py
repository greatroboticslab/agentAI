"""The targets and the query grammar of discovery (docs/CONTINUOUS_LOOP.md §7.1).

Queries are generated from configuration only; nothing here is a seed list:
  * the funnel domain's targets (taxon, genus, common name, rank);
  * the collector config's search_terms (the owner's table of common names)
    and priority;
  * the authority's vernacular answers held by the taxonomy cache
    (names.Names.vernacular_names);
  * the project's alias table, which the config names by module attribute
    (config.alias_table), for matching class names and text;
  * the EPPO table the config pins (codes of the targets, reverse lookup).

Templates (config query_grammar.templates) are per kind: "text" (free-text
search of dataset repositories), "class" (a class-name search), "taxon" (a
binomial search of an annotation index) and "eppo" (EPPO codes). Fields:
{binomial} (the target taxon, a genus for a genus-rank target), {genus},
{common} (each common name) and {eppo} (each EPPO code). Queries interleave
the targets (round robin, deficit classes first, then config priority), so a
provider's query cap covers every requested target.

mentioned(text) finds the targets a text names: a phrase (the target's name
split at capitals, its common names, its taxon, its search terms and
vernacular names) as whole words, or an alias key as a whole token.
"""
from __future__ import annotations

import re

from . import ConfigError
from .config import alias_table

_WS = re.compile(r"[^a-z0-9]+")
_TOKEN = re.compile(r"[a-z0-9]+")


def norm_text(t):
    return " %s " % _WS.sub(" ", str(t or "").lower()).strip()


def camel_words(name):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(name or ""))


def name_key(t):
    return re.sub(r"[^a-z0-9]", "", str(t or "").lower())


class Targets(object):
    def __init__(self, cfg, names=None):
        self.cfg = cfg
        self.list = cfg.targets
        self.by_name = {t["name"]: t for t in self.list}
        self.aliases = alias_table(cfg)
        self.search_terms = {k: list(v) for k, v in (cfg.raw.get("search_terms") or {}).items()}
        self.priority = dict(cfg.raw.get("priority") or {})
        self.vernacular = names.vernacular_names() if names is not None else {}
        self._phrases = None

    # ---- ordering
    def ordered(self, deficit=None):
        """Target names: the deficit classes first (in the order given), then the
        rest by config priority (1 first), then id. With a deficit list, only
        those classes."""
        if deficit:
            bad = [d for d in deficit if d not in self.by_name]
            if bad:
                raise ConfigError("--classes names non-targets %s (targets: %s)" % (bad, sorted(self.by_name)))
            return list(dict.fromkeys(deficit))
        return [t["name"] for t in sorted(self.list, key=lambda t: (self.priority.get(t["name"], 99), t["id"]))]

    # ---- names of a target
    def commons(self, tname):
        t = self.by_name[tname]
        out = []
        for c in [t.get("common")] + self.search_terms.get(tname, []) + self.vernacular.get(tname, []):
            if isinstance(c, str) and c.strip() and c.strip().lower() not in [o.lower() for o in out]:
                out.append(c.strip())
        return out

    def eppo_codes(self, tname):
        t = self.by_name[tname]
        return self.cfg.eppo_codes_of(t.get("taxon") or "")

    def fields(self, tname):
        t = self.by_name[tname]
        taxon = t.get("taxon") or ""
        genus = t.get("genus") or taxon.split(" ")[0]
        return {"binomial": taxon, "genus": genus, "common": self.commons(tname), "eppo": self.eppo_codes(tname)}

    # ---- queries
    def queries(self, kind, deficit=None, limit=None, templates=None):
        """[(query, target name)] of the templates of `kind` (or the given
        templates: a provider's own list), targets interleaved round robin,
        case-insensitively unique."""
        tpls = list(templates) if templates else ((self.cfg.raw["query_grammar"]["templates"] or {}).get(kind) or [])
        per = []
        for tname in self.ordered(deficit):
            f = self.fields(tname)
            qs = []
            for tpl in tpls:
                vals = [{}]
                for fld in ("common", "eppo"):
                    if "{%s}" % fld in tpl:
                        vals = [dict(v, **{fld: x}) for v in vals for x in f[fld]]
                for v in vals:
                    try:
                        q = tpl.format(binomial=f["binomial"], genus=f["genus"], common=v.get("common", ""),
                                       eppo=v.get("eppo", ""))
                    except (KeyError, IndexError) as e:
                        raise ConfigError("query template %r: %s" % (tpl, e))
                    q = q.strip()
                    if q and q.strip('"').strip():
                        qs.append(q)
            per.append((tname, qs))
        out, seen = [], set()
        i = 0
        while any(i < len(qs) for _t, qs in per):
            for tname, qs in per:
                if i < len(qs) and qs[i].lower() not in seen:
                    seen.add(qs[i].lower())
                    out.append((qs[i], tname))
            i += 1
        return out[:limit] if limit else out

    # ---- matching
    def phrases(self):
        if self._phrases is None:
            ph = {}
            for t in self.list:
                n = t["name"]
                cands = [camel_words(n), t.get("taxon") or ""] + self.commons(n)
                ph[n] = sorted({norm_text(c) for c in cands if norm_text(c).strip()})
            self._phrases = ph
        return self._phrases

    def mentioned(self, text):
        """Target names the text names (module docstring)."""
        nt = norm_text(text)
        toks = set(_TOKEN.findall(nt))
        out = set()
        for n, ps in self.phrases().items():
            if any(p in nt for p in ps):
                out.add(n)
        for tk in toks:
            n = self.aliases.get(tk) or (self.aliases.get(tk[:-1]) if tk.endswith("s") else None)
            if n:
                out.add(n)
        return sorted(out, key=lambda n: self.by_name[n]["id"])

    def alias_target(self, name):
        """The target a class name is by the alias table (name key, a plural
        s dropped), else None."""
        k = name_key(name)
        if k in self.aliases:
            return self.aliases[k]
        if k.endswith("s") and k[:-1] in self.aliases:
            return self.aliases[k[:-1]]
        return None
