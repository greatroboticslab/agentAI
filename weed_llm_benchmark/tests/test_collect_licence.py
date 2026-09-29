#!/usr/bin/env python3
"""The licence gate (docs/CONTINUOUS_LOOP.md §3.2 "Licence", §2.5 P6, §7.5, §8
"Licences"; group D acceptance "Licence: gate cases per P6").

Pinned:
  * one canonical id for the forms providers use (Zenodo ids, Mendeley short
    names, Kaggle licence names, Creative Commons URLs, a record server's
    "by, <url>", SPDX ids, "CC0: Public Domain", the ODbL wording);
  * P6 through the config's policy: permissive -> usable; non-commercial and
    no-derivatives -> research_only (every image flagged); unresolved (no
    licence, "other", an unknown text) -> held; "all rights reserved" ->
    refused (the source is closed);
  * the preferred copy of one dataset: permissive before research-only
    before unresolved;
  * the fallback asks license_audit.detect_license (injected here: no
    network) and records its answer as evidence;
  * the collector never uploads: no upload or sync call anywhere in its code.

Run:  python3 tests/test_collect_licence.py
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_collect_world as W  # noqa: E402

TMP = W.setup("collect_licence_")
check, raises = W.check, W.raises


def main():
    try:
        from weed_optimizer_framework.tools.collect import licence as L
        cfg = W.config()
        pol = cfg.raw["licence_policy"]
        print("canonical ids")
        cases = {
            "cc-by-4.0": "cc-by-4.0", "CC BY 4.0": "cc-by-4.0", "CC-BY-4.0": "cc-by-4.0",
            "https://creativecommons.org/licenses/by/4.0/": "cc-by-4.0",
            "by, http://creativecommons.org/licenses/by/4.0": "cc-by-4.0",
            "Attribution 4.0 International (CC BY 4.0)": "cc-by-4.0",
            "CC BY-NC-SA 4.0": "cc-by-nc-sa-4.0", "cc-by-nc-sa-4.0": "cc-by-nc-sa-4.0",
            "https://creativecommons.org/licenses/by-nc/3.0/": "cc-by-nc-3.0",
            "CC BY-ND 4.0": "cc-by-nd-4.0", "CC0: Public Domain": "cc0", "cc0-1.0": "cc0",
            "Public Domain": "public-domain", "MIT": "mit", "Apache-2.0": "apache-2.0", "GPL-3.0": "gpl-3.0",
            "Database: Open Database, Contents: Database Contents": "odbl", "NOASSERTION": "unresolved",
            "Other (specified in description)": "unresolved", None: "unresolved", "": "unresolved",
            "unreachable": "unresolved", "All rights reserved": "all-rights-reserved",
            "Some bespoke terms": "other:some-bespoke-terms",
            "CC BY Non Commercial": "cc-by-nc", "Free for research only": "research-only",
            "MIT, for academic use only": "research-only", "Non-profit use": "research-only",
        }
        bad = {k: (L.canonical(k), v) for k, v in cases.items() if L.canonical(k) != v}
        check("every provider form maps to one canonical id", not bad, bad)
        print("P6 classes")
        g = lambda t: L.gate(t, pol, evidence={"provider": "x"})  # noqa: E731
        check("CC BY is permissive, not research-only", g("CC BY 4.0")["class"] == "permissive"
              and not g("CC BY 4.0")["research_only"])
        check("CC BY-SA, CC0, MIT, ODbL are permissive", all(g(t)["class"] == "permissive" for t in
                                                            ("CC BY-SA 4.0", "CC0: Public Domain", "MIT", "ODbL")))
        check("non-commercial licences are research-only", all(g(t)["class"] == "research_only"
                                                               and g(t)["research_only"] for t in
                                                               ("CC BY-NC 4.0", "CC BY-NC-SA 4.0", "CC BY-NC-ND 4.0")))
        check("no-derivatives is research-only", g("CC BY-ND 4.0")["class"] == "research_only")
        check("a restriction word wins over a permissive name (research, academic, non-profit use)",
              all(g(t)["class"] == "research_only" for t in ("CC BY Non Commercial", "Free for research only",
                                                             "MIT, for academic use only")))
        restricted = ("CC BY 4.0 (research use only)", "CC BY-SA 4.0, for academic use only", "ODbL, research only",
                      "Public domain, for educational purposes only", "CC0 personal use only",
                      "https://creativecommons.org/licenses/by/4.0/ non-commercial research",
                      "Creative Commons Attribution, non-profit use", "Apache-2.0, evaluation only")
        check("a restriction wins over every permissive family (CC BY and BY-SA, CC0, public domain, ODbL, Apache)",
              all(g(t)["class"] == "research_only" and g(t)["research_only"] for t in restricted),
              {t: g(t)["id"] for t in restricted if g(t)["class"] != "research_only"})
        try:
            from weed_optimizer_framework.tools.inc2 import splits as SP
        except ImportError as e:
            print("  skip agreement with inc2.splits.research_only: not importable (%s)" % e)
        else:
            texts = list(restricted) + ["CC BY 4.0", "CC BY-SA 4.0", "CC0: Public Domain", "MIT", "Apache-2.0",
                                        "CC BY-NC 4.0", "CC BY-NC-SA 4.0", "CC BY Non Commercial",
                                        "Free for research only", "MIT, for academic use only"]
            dis = {t: (g(t)["class"], SP.research_only(t)) for t in texts
                   if (g(t)["class"] == "research_only") != SP.research_only(t)}
            check("the collector and inc2.splits read a licence's restriction alike", not dis, dis)
        check("no licence, 'other' and an unknown text are unresolved (held)",
              all(g(t)["class"] == "unresolved" for t in (None, "other", "Some bespoke terms")))
        check("all rights reserved is refused", g("All rights reserved")["class"] == "refused")
        check("the gate keeps the text and the evidence", g("CC BY 4.0")["text"] == "CC BY 4.0"
              and g("CC BY 4.0")["evidence"] == {"provider": "x"})
        recs = [g("CC BY-NC 4.0"), g(None), g("CC BY 4.0")]
        check("the preferred copy is the permissive one", L.prefer(recs) == 2)
        check("... then research-only before unresolved", L.prefer(recs[:2]) == 0)
        print("fallback")
        seen = []

        def detect(slug, info):
            seen.append(slug)
            return {"license": "CC BY 4.0", "license_source": "kaggle:datasets/view"}
        fb = L.fallback("kg_a__b", {}, detect=detect)
        check("the fallback asks license_audit.detect_license and records its answer",
              seen == ["kg_a__b"] and fb["text"] == "CC BY 4.0" and fb["evidence"]["license_source"]
              == "kaggle:datasets/view", fb)
        fb2 = L.fallback("kg_a__b", {}, detect=lambda s, i: {"license": "unreachable", "license_source": "kaggle:no-token"})
        check("an unreachable answer gives no licence text (held)", fb2["text"] is None)
        print("nothing is redistributed")
        col = W.ROOT / "weed_optimizer_framework" / "tools" / "collect"
        pat = re.compile(r"(\.upload\(|upload_image|roboflow_sync|AUTO_SYNC|single_upload|\bupload_dataset\b|"
                         r"method=\"(PUT|DELETE)\")")
        hits = []
        for p in sorted(col.rglob("*.py")):
            for i, line in enumerate(p.read_text().splitlines(), 1):
                if pat.search(line):
                    hits.append((p.name, i, line.strip()))
        check("no upload, sync, PUT or DELETE call in the collector's code", not hits, hits)
    finally:
        W.cleanup()
    print("\n%d failure(s)" % len(W.FAILURES))
    return 1 if W.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
