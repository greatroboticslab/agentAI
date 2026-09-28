#!/usr/bin/env python3
"""The funnel engine is domain-free (docs/FUNNEL_AUDIT.md 8.8; runner 7.5).

Files: every .py under weed_optimizer_framework/tools/funnel/ except
adapters/ and domains/ (where domain knowledge belongs).

Normalisation: the text is lowercased, and every occurrence of
weed_optimizer_framework and weed_llm_benchmark (the code's own location) is
removed.

Terms, from every config in funnel/domains/*.json and from the R13 vehicles
fixture, through funnel.domain.Domain.terms():
  * substrings (case-insensitive): each config's domain_terms and every
    source slug it names, plus the fixed cross-domain list below;
  * whole tokens (a maximal [a-z0-9]+ run of the normalised text): the
    domain name, every class name without its non-alphanumerics, every word
    of five or more letters of a target, 'not', attractor or KT7 taxon,
    every exam name but dev and test, and every lab group name;
  * the common names of the targets and attractors (contract 8.8: "a species
    name"; runner 7.5 lists only the taxon words): a one-word common name is a
    whole token, a longer one a lowercase phrase (a substring), so a
    vernacular name such as an attractor's cannot hide in an engine comment.

Planted checks: three temporary files in a copy of the engine, holding
"Amaranthus", "OtherPlant" and "indicates": the first two are flagged, and
"indicates" is not flagged by the taxon word "indica" (whole tokens only).

Coupling outside the engine (inc_autopilot/model.py DOMAIN, the brain_plan
prompt header, verify.CWD12_RELATED_TOKENS, the D2 summary strings) is out
of this test's scope and listed in the contract as coupling to migrate; the
test-blindness exam list, which the contract names too, now comes from the
domain config (tests/test_funnel_ap_replay.py, R13).

Run:  python3 tests/test_funnel_domain_free.py
"""
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile

TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(ROOT))
FUN = ROOT / "weed_optimizer_framework" / "tools" / "funnel"
R13 = TESTS / "fixtures" / "inc_replay" / "funnel" / "vehicles" / "vehicles.json"
FIXED = ("cwd12", "otherplant", "bioclip", "ndsu", "cottonweed")
STRIP = ("weed_optimizer_framework", "weed_llm_benchmark")
TOKEN_RE = re.compile(r"[a-z0-9]+")
FAILURES, SKIPS = [], []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:2000]))
        FAILURES.append(name)


def configs():
    """Every domain config the grep reads: funnel/domains/*.json and the R13 fixture."""
    out = sorted(str(p) for p in (FUN / "domains").glob("*.json"))
    if R13.is_file():
        out.append(str(R13))
    return out


def common_names():
    """(tokens, phrases): the targets' and attractors' common names of every
    config, a one-word name as a whole token, a longer one as a phrase."""
    tokens, phrases = set(), set()
    for c in configs():
        raw = json.loads(pathlib.Path(c).read_text(encoding="utf-8"))
        names = [t.get("common") for t in (raw.get("classes") or {}).get("targets") or []]
        names += [a.get("common") for a in raw.get("attractors") or []]
        for n in names:
            if not isinstance(n, str) or not n.strip():
                continue
            norm = " ".join(n.lower().split())
            words = TOKEN_RE.findall(norm)
            if len(words) == 1:
                tokens.add(words[0])
            elif words:
                phrases.add(norm)
    return tokens, phrases


def terms():
    """{"substring": [...], "token": [...]}: the union over every config, plus
    FIXED and the common names (common_names)."""
    from weed_optimizer_framework.tools.funnel import domain as FD
    sub, tok = set(FIXED), set()
    for c in configs():
        t = FD.load(c).terms()
        sub.update(x.lower() for x in t["substring"] if x)
        tok.update(x.lower() for x in t["token"] if x)
    ctok, cphr = common_names()
    tok |= ctok
    sub |= cphr
    return {"substring": sorted(sub), "token": sorted(tok)}


def config_tokens():
    """The whole tokens funnel.domain.Domain.terms() gives (class names, taxon
    words, ...), so the planted common-name check picks a name only the
    common-name rule catches."""
    from weed_optimizer_framework.tools.funnel import domain as FD
    out = set()
    for c in configs():
        out.update(x.lower() for x in FD.load(c).terms()["token"] if x)
    return out


def engine_files(root=None):
    root = pathlib.Path(root or FUN)
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts
                  and p.relative_to(root).parts[0] not in ("adapters", "domains"))


def normalise(text):
    t = text.lower()
    for s in STRIP:
        t = t.replace(s, "")
    return t


def scan(root=None, t=None):
    """[(file, line, term)] of every domain term in the engine's files."""
    t = t or terms()
    toks = set(t["token"])
    out = []
    for p in engine_files(root):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            n = normalise(line)
            for s in t["substring"]:
                if s in n:
                    out.append((str(p.relative_to(pathlib.Path(root or FUN))), i, s))
            for w in TOKEN_RE.findall(n):
                if w in toks:
                    out.append((str(p.relative_to(pathlib.Path(root or FUN))), i, w))
    return out


def main():
    print("the funnel engine names no domain term")
    try:
        t = terms()
    except Exception as e:
        check("every domain config loads", False, "%s: %s" % (type(e).__name__, e))
        print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
        return 1
    check("the terms come from %d config(s), including the R13 vehicles fixture" % len(configs()),
          len(configs()) >= 2 and str(R13) in configs() and "vehicles" in t["token"] and "car" in t["token"],
          configs())
    check("the weed config's terms include its classes, taxa and exams",
          {"ragweed", "otherplant"} <= set(t["token"]) | set(t["substring"]) and "ood22" in t["token"]
          and "amaranthus" in t["token"], [x for x in t["token"] if x.startswith("am")])
    files = engine_files()
    check("the engine has files to scan (%d)" % len(files), len(files) >= 10, [str(p) for p in files])
    problems = scan(t=t)
    check("no engine file outside adapters/ and domains/ holds a domain term", not problems, problems[:20])
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="funnel_domain_free_"))
    try:
        copy = tmp / "funnel"
        shutil.copytree(str(FUN), str(copy), ignore=shutil.ignore_patterns("__pycache__"))
        (copy / "planted_taxon.py").write_text("# a planted line: Amaranthus palmeri\n")
        (copy / "planted_class.py").write_text("LABEL = 'OtherPlant'\n")
        (copy / "planted_word.py").write_text("# this indicates nothing about a domain\n")
        found = scan(copy, t)
        by = {}
        for f, _i, term in found:
            by.setdefault(f, []).append(term)
        check("a planted 'Amaranthus' is flagged", "amaranthus" in by.get("planted_taxon.py", []), by)
        check("a planted 'OtherPlant' is flagged", "otherplant" in by.get("planted_class.py", []), by)
        check("'indicates' is not flagged by the taxon word 'indica' (whole tokens only)",
              "planted_word.py" not in by and "indica" in t["token"], by.get("planted_word.py"))
        ctok, cphr = common_names()
        one = sorted(ctok - config_tokens())[:1]
        (copy / "planted_common.py").write_text("# a relative such as a %s\n# a %s\n"
                                                % ((one or ["?"])[0], sorted(cphr)[0] if cphr else "?"))
        by2 = {}
        for f, _i, term in scan(copy, t):
            by2.setdefault(f, []).append(term)
        check("a planted one-word common name (%s) and a planted multi-word one (%s) are flagged"
              % ((one or ["none"])[0], sorted(cphr)[0] if cphr else "none"),
              one and cphr and one[0] in by2.get("planted_common.py", [])
              and sorted(cphr)[0] in by2.get("planted_common.py", []), by2.get("planted_common.py"))
        (copy / "adapters" / "planted_ok.py").write_text("# Amaranthus belongs here\n")
        check("adapters/ is outside the scan", "adapters/planted_ok.py" not in {f for f, _i, _t in scan(copy, t)})
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
