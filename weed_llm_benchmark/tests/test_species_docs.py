#!/usr/bin/env python3
"""The documents that name cwd12 species agree with cwd12_species.py.

docs/CWD12_SPECIES.md carries the id -> species table in prose; the results
page lists the holdout's species; make_figures names per-species rows. Each
must say what the code says, and the per-species table must be translated from
the old labels its source recorded, not copied under them.

Run:  python3 tests/test_species_docs.py
"""
import json
import pathlib
import re
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parents[1]
GIT_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from weed_optimizer_framework.tools import cwd12_species as S  # noqa: E402
import make_figures as MF  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def main():
    doc = (GIT_ROOT / "docs" / "CWD12_SPECIES.md").read_text()

    print("docs/CWD12_SPECIES.md table")
    rows = re.findall(r"^\| (\d+) \| (\w+) \| \*\*([^*]+)\*\* \| (\d+) \| \*([^|]+?)\*( spp\.)? \|$",
                      doc, re.M)
    check("twelve id rows", len(rows) == 12, len(rows))
    for cid, legacy, common, slot, binom, spp in rows:
        i = int(cid)
        sp = S.CWD12_SPECIES[i]
        check("id %d legacy label" % i, legacy == S.CWD12_LEGACY_LABELS[i], legacy)
        check("id %d species" % i, common == S.CWD12_COMMON[sp], common)
        check("id %d trainer slot" % i, int(slot) == S.CWD12_ID_TO_SLOT[i], slot)
        check("id %d binomial" % i, (binom + (spp or "")) == S.CWD12_BINOMIAL[sp], binom)
        check("id %d slot holds the same species" % i,
              S.TRAINER_SLOT_SPECIES[int(slot)] == sp)

    print("results page")
    page = (GIT_ROOT / "docs" / "results_page" / "index.html").read_text()
    stamp = re.search(r"across 12 species(.*?)</p>", page, re.S)
    listed = [x.strip() for x in stamp.group(1).split("·") if x.strip()] if stamp else []
    check("holdout lists the twelve species",
          sorted(listed) == sorted(S.CWD12_COMMON.values()), listed)
    for bad in ("Crabgrass", "Nutsedge", "Carpetweeds", "Morningglory"):
        check("results page does not name %s" % bad, bad not in page)

    print("correction notes")
    for rel in ("docs/RESULTS_TABLE.md", "docs/BEST_MODEL_CARD.md",
                "docs/SUPERWEED_PLAN.md", "docs/SCIENCE_AUDIT.md"):
        txt = (GIT_ROOT / rel).read_text()
        check("%s carries a dated note linking CWD12_SPECIES.md" % rel,
              "Correction 2026-09-21" in txt and "(CWD12_SPECIES.md)" in txt)
    card = (GIT_ROOT / "docs" / "BEST_MODEL_CARD.md").read_text()
    for legacy, value in (("Ragweed", "0.9767"), ("Morningglory", "0.7324"),
                          ("Nutsedge", "0.8585"), ("Goosegrass", "0.7973")):
        sp = S.CWD12_COMMON[S.legacy_to_species(legacy)]
        check("card note: %s %s reads as %s" % (legacy, value, sp),
              "| %s | %s | %s" % (legacy, sp, value) in card)
    readme = (HERE / "README.md").read_text()
    link = re.search(r"\]\((\.\./docs/CWD12_SPECIES\.md)\)", readme)
    check("README link resolves", bool(link) and (HERE / link.group(1)).resolve().is_file())
    log = (GIT_ROOT / "RESEARCH_LOG.md").read_text()
    # The correction must stay in the log; later entries go above it.
    heads = re.findall(r"^## .*$", log, re.M)
    corr = [h for h in heads if h.startswith("## 2026-09-21") and "cwd12 class names were wrong" in h]
    check("RESEARCH_LOG holds the 2026-09-21 correction entry", len(corr) == 1, heads[:3])

    print("make_figures.species_rows")
    legacy_rows = [{"cls": n, "map50": 0.5, "map50_95": 0.4} for n in S.CWD12_LEGACY_LABELS]
    out, translated = MF.species_rows(legacy_rows)
    check("a legacy list is translated", translated)
    check("rows are named by species",
          [r["species"] for r in out] == [S.CWD12_COMMON[s] for s in S.CWD12_SPECIES])
    check("the source label is kept", [r["label"] for r in out] == S.CWD12_LEGACY_LABELS)
    check("values are untouched", all(r["map50_95"] == 0.4 for r in out))
    sp_rows = [{"cls": n, "map50": 0.5, "map50_95": 0.4} for n in S.CWD12_SPECIES]
    out2, translated2 = MF.species_rows(sp_rows)
    check("a species list is not translated again", not translated2)
    check("species rows keep their species", [r["species"] for r in out2]
          == [S.CWD12_COMMON[s] for s in S.CWD12_SPECIES])
    part = [{"cls": "Ragweed", "map50": 0.1, "map50_95": 0.1}]
    out3, translated3 = MF.species_rows(part)
    check("a lone 'Ragweed' is not read as a legacy label",
          not translated3 and out3[0]["species"] == "Ragweed")

    print("per_species.md is make_figures output")
    d = json.loads(MF.DATA.read_text())
    check("source rows are the legacy list",
          [r["cls"] for r in d["per_species_yolo11n_val"]["rows"]] == S.CWD12_LEGACY_LABELS)
    with tempfile.TemporaryDirectory() as td:
        old_out = MF.OUT
        try:
            MF.OUT = pathlib.Path(td)
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                MF.main()
            fresh = (MF.OUT / "per_species.md").read_text()
        except ImportError as e:     # matplotlib missing: compare the table only
            print("  (skipped full regeneration: %s)" % e)
            fresh = None
        finally:
            MF.OUT = old_out
    committed = (HERE / "results" / "figures" / "per_species.md").read_text()
    if fresh is not None:
        check("committed per_species.md equals a fresh run", committed == fresh)
    check("per_species.md names Sicklepod beside its 0.983",
          "| Sicklepod | Ragweed | 0.993 | 0.983 |" in committed)

    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
