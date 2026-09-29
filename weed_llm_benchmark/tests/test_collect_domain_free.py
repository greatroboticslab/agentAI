#!/usr/bin/env python3
"""The collector is domain-free (docs/CONTINUOUS_LOOP.md §6.8 S13, §9 group D):
everything domain-specific sits in collect/domains/<domain>.json and the funnel
domain config it points to; the code under tools/collect/ names none of it.

Files: every .py under weed_optimizer_framework/tools/collect/ (domains/ holds
only JSON).

Normalisation (as tests/test_funnel_domain_free.py): lower case; every
occurrence of the code's own location (weed_optimizer_framework,
weed_llm_benchmark) removed.

Terms:
  * the funnel domain-free test's terms, read through its own functions: every
    funnel domain config and the R13 vehicles fixture (Domain.terms()), the
    fixed cross-domain list and the targets' and attractors' common names;
  * from every collector config (collect/domains/*.json): substrings = the
    known items' ids, refs, source ids and the literal refs of their match
    rules, the card-tabled source ids and category names, the
    hosts of the providers' base_url and FTP host, and multi-word search
    terms; whole tokens = the lab group names, the binomial words (five or
    more letters) of the pinned EPPO table and one-word search terms; the
    EPPO codes themselves as upper-case whole tokens of the raw line (a code
    such as MATCH is an English word in lower case).

Planted checks: a copy of the package with planted lines holding an EPPO
code, a lab group name, a known item's ref and a host are each flagged; a
word that merely contains a short term is not (whole tokens only).

Run:  python3 tests/test_collect_domain_free.py
"""
import json
import pathlib
import re
import shutil
import sys
import tempfile

TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))
COL = ROOT / "weed_optimizer_framework" / "tools" / "collect"
TOOLS = COL.parent
FAILURES = []

import test_funnel_domain_free as FDF  # noqa: E402  (its terms(), normalise() and TOKEN_RE)


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:2000]))
        FAILURES.append(name)


def collect_configs():
    out = []
    for p in sorted((COL / "domains").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("format") == "collect-domain/1":
            out.append((p, d))
    return out


def collect_terms():
    sub, tok, upper = set(), set(), set()
    for _p, d in collect_configs():
        for it in d.get("known_items") or []:
            for k in ("id", "name", "ref"):
                if it.get(k):
                    sub.add(str(it[k]).lower())
            for m in it.get("match") or []:
                if m.get("ref"):
                    sub.add(str(m["ref"]).lower())
        for sid, t in (d.get("card_class_tables") or {}).items():
            sub.add(sid.lower())
            for n in (t.get("by_name") or {}):
                sub.add(n.lower())
        for sec in (d.get("providers") or {}).values():
            for u in (sec.get("base_url"), (sec.get("ftp") or {}).get("host")):
                if u:
                    sub.add(re.sub(r"^[a-z]+://", "", u.lower()).rstrip("/"))
        for g in (d.get("lab_groups") or {}):
            tok.add(g.lower())
        for terms in (d.get("search_terms") or {}).values():
            for t in terms:
                words = re.findall(r"[a-z0-9]+", t.lower())
                if len(words) == 1:
                    tok.add(words[0])
                elif words:
                    sub.add(" ".join(words))
        ep = d.get("eppo")
        if ep:
            e = json.loads((TOOLS / ep["path"]).read_text(encoding="utf-8"))
            for code, name in e["codes"].items():
                upper.add(code)
                for w in re.findall(r"[a-z]{5,}", name.lower()):
                    tok.add(w)
    return {"substring": sorted(sub), "token": sorted(tok), "upper": sorted(upper)}


def all_terms():
    t = FDF.terms()
    c = collect_terms()
    return {"substring": sorted(set(t["substring"]) | set(c["substring"])),
            "token": sorted(set(t["token"]) | set(c["token"])), "upper": c["upper"]}


def code_files(root=None):
    root = pathlib.Path(root or COL)
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts
                  and p.relative_to(root).parts[0] != "domains")


def scan(root=None, t=None):
    t = t or all_terms()
    toks = set(t["token"])
    upper = set(t.get("upper") or ())
    out = []
    root = pathlib.Path(root or COL)
    for p in code_files(root):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for w in re.findall(r"[A-Z0-9]+", line):
                if w in upper:
                    out.append((str(p.relative_to(root)), i, w))
            n = FDF.normalise(line)
            for s in t["substring"]:
                if s in n:
                    out.append((str(p.relative_to(root)), i, s))
            for w in FDF.TOKEN_RE.findall(n):
                if w in toks:
                    out.append((str(p.relative_to(root)), i, w))
    return out


def main():
    print("the collector names no domain term")
    t = all_terms()
    c = collect_terms()
    check("the collector configs give terms (%d substrings, %d tokens)" % (len(c["substring"]), len(c["token"])),
          len(collect_configs()) >= 1 and "tamu" in c["token"] and "POROL" in c["upper"]
          and any("yuzhenlu" in s for s in c["substring"]), c)
    check("the funnel's terms (both configs) are included", {"otherplant", "car", "amaranthus"} <= set(t["token"])
          | set(t["substring"]))
    files = code_files()
    check("the package has files to scan (%d)" % len(files), len(files) >= 15, [str(p) for p in files])
    problems = scan(t=t)
    check("no collector code file holds a domain term", not problems, problems[:30])
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="collect_domain_free_"))
    try:
        copy = tmp / "collect"
        shutil.copytree(str(COL), str(copy), ignore=shutil.ignore_patterns("__pycache__"))
        (copy / "planted_eppo.py").write_text("CODE = 'POROL'\n")
        (copy / "planted_lab.py").write_text("# the TAMU group\n")
        (copy / "planted_ref.py").write_text("REF = 'yuzhenlu/cottonweeddet3'\n")
        (copy / "planted_host.py").write_text("URL = 'https://dataserv.ub.tum.de/'\n")
        (copy / "planted_ok.py").write_text("# a carousel of buses, a match and a porolith are fine\n")
        by = {}
        for f, _i, term in scan(copy, t):
            by.setdefault(f, []).append(term)
        check("a planted EPPO code is flagged (EPPO codes match upper case, whole tokens)",
              "POROL" in by.get("planted_eppo.py", []), by)
        check("a planted lab group name is flagged", "tamu" in by.get("planted_lab.py", []), by)
        check("a planted known-item ref is flagged", bool(by.get("planted_ref.py")), by)
        check("a planted provider host is flagged", bool(by.get("planted_host.py")), by)
        check("words that merely contain a short term are not flagged (whole tokens)",
              "planted_ok.py" not in by, by.get("planted_ok.py"))
        (copy / "domains" / "planted.py").write_text("# POROL belongs in the domain directory\n")
        check("domains/ is outside the scan", not any(f.startswith("domains") for f, _i, _t in scan(copy, t)))
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
