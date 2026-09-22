#!/usr/bin/env python3
"""A harvested box joins a cwd12 slot only if it names the species that slot holds.

The trainer's slots 0-11 were named from the alphabetical list in cwd12's
data.yaml, which is not the dataset's class list (cwd12_species.py has the
evidence). Joining external class names through those labels sent
3SeasonWeedDet10's Purslane into the slot that holds Palmer amaranth, its
Ragweed into Sicklepod, a Roboflow set's Crabgrass and Nutsedge into Morning
glory and Ragweed, and deleted every Waterhemp, Carpetweed and Morning glory
box it could not name. One cwd12 copy (leave4out/dataset_holdout) was read the
same way although its files use the original ids.

These pin what replaced it: the slot ids do not move; each slot's species is
fixed by the permutation cwd12 already goes through; names join by species with
aliases; unknown names go to an aux slot instead of being deleted; and the two
cwd12 copies are mapped by id space and re-checked against cwd12's own labels.
Re-encoded copies (within 3 dHash bits) of the sealed holdout or of a verified
cwd12 photo are blocked; other near matches are not, since padded and studio
images of different plants also fall within 3 bits.

Run:  python3 tests/test_cwd12_species.py
"""
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools import cwd12_species as S  # noqa: E402
from weed_optimizer_framework.tools import mega_trainer as M  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def slot_species(ds_map):
    """source id -> species name, or 'aux' for slots 12-99."""
    return {i: (M.CANONICAL_12_SPECIES[c] if 0 <= c < 12 else "aux")
            for i, c in ds_map.items()}


def main():
    # --- the slot space itself ------------------------------------------------
    check("slot ids are unchanged (checkpoints stay valid)",
          M.CWD12_ORIG_TO_CANON == {0: 0, 1: 1, 2: 8, 3: 9, 4: 10, 5: 11,
                                    6: 2, 7: 3, 8: 4, 9: 5, 10: 6, 11: 7},
          M.CWD12_ORIG_TO_CANON)
    check("each slot holds the species of the cwd12 id mapped into it",
          all(M.CANONICAL_12_SPECIES[M.CWD12_ORIG_TO_CANON[i]] == sp
              for i, sp in enumerate(S.CWD12_SPECIES)))
    check("the twelve slots are twelve different species",
          sorted(M.CANONICAL_12_SPECIES) == sorted(S.CWD12_SPECIES))
    check("cwd12 has no Crabgrass and no Nutsedge",
          not {"Crabgrass", "Nutsedge"} & set(S.CWD12_SPECIES))

    # --- name matching --------------------------------------------------------
    for name, want in (("Carpet weed", "Carpetweed"), ("Carpetweeds", "Carpetweed"),
                       ("Morning glory", "MorningGlory"), ("morning_glory", "MorningGlory"),
                       ("Palmer Amaranth", "PalmerAmaranth"), ("Waterhemp", "Waterhemp"),
                       ("Amaranthus palmeri", "PalmerAmaranth"),
                       ("Cutleaf Groundcherry", "CutleafGroundcherry"),
                       ("Crabgrass", None), ("Nutsedge", None), ("Giant ragweed", None),
                       ("Lambsquarters", None), ("Redroot Pigweed", None), ("Corn", None)):
        check("%r names %s" % (name, want or "no cwd12 species"),
              S.species_of(name) == want, "got %r" % S.species_of(name))

    # cli_species: species keys and aliases, never a legacy-only label
    for name, want in (("SpottedSpurge", "SpottedSpurge"), ("Spotted spurge", "SpottedSpurge"),
                       ("Goosegrass", "Goosegrass"), ("Carpetweed", "Carpetweed"),
                       ("Carpetweeds", None), ("Morningglory", None), ("Crabgrass", None),
                       ("Nutsedge", None), ("Morning glory", "MorningGlory"), ("Corn", None)):
        check("cli_species(%r) is %r" % (name, want), S.cli_species(name) == want,
              "got %r" % S.cli_species(name))

    # --- provenance: which lists are legacy, and translating them once -------
    check("the data.yaml list is a legacy list", S.is_legacy_label_list(S.CWD12_LEGACY_LABELS))
    check("the trainer slot list is a legacy list", S.is_legacy_label_list(S.TRAINER_SLOT_LEGACY))
    legacy_head = {i: n for i, n in enumerate(S.TRAINER_SLOT_LEGACY)}
    legacy_head.update({i: "aux_%d" % i for i in range(12, 100)})
    check("an old 100-class head (12 legacy + aux) is a legacy list", S.is_legacy_label_list(legacy_head))
    check("a species list is not a legacy list", not S.is_legacy_label_list(S.CWD12_SPECIES))
    check("a new head (12 species + aux) is not a legacy list",
          not S.is_legacy_label_list(S.TRAINER_SLOT_SPECIES + ["aux_%d" % i for i in range(12, 100)]))
    check("a CottonWeedID15 copy's real names are not a legacy list",
          not S.is_legacy_label_list(["Carpet weed", "Crabgrass", "Eclipta", "Goosegrass",
                                      "Morning glory", "Nutsedge"]))
    # the lab's results/leave4out/yolo_augmented/train/weights/best.pt model.names
    aug = {0: "Carpetweeds", 1: "Crabgrass", 2: "PalmerAmaranth", 3: "PricklySida",
           4: "Purslane", 5: "Ragweed", 6: "Sicklepod", 7: "SpottedSpurge",
           8: "novel_weed_llm"}
    check("a leave-4-out head (sp8 legacy + novel) is a legacy list",
          S.is_legacy_label_list(aug))
    check("the pre-v3.60.0 Config.get_species_names() list is a legacy list",
          S.is_legacy_label_list(S.TRAINER_SLOT_LEGACY[:8] + ["novel_weed"]))
    check("a leave-4-out head translates, novel class kept",
          S.species_names_for(aug) == dict(enumerate(S.TRAINER_SLOT_SPECIES[:8] + ["novel_weed_llm"])),
          S.species_names_for(aug))
    check("leave-4-out head id 4 'Purslane' is PalmerAmaranth",
          S.class_species("x", 4, aug) == "PalmerAmaranth" and S.class_species("x", 8, aug) is None)
    check("sp8 legacy + a real species name is not a legacy list",
          not S.is_legacy_label_list(S.TRAINER_SLOT_LEGACY[:8] + ["Waterhemp"]))
    tr = S.species_names_for(legacy_head)
    check("an old head's names translate slot by slot",
          [tr[i] for i in range(12)] == S.TRAINER_SLOT_SPECIES and tr[57] == "aux_57", tr)
    check("translating a translated list changes nothing (no double translation)",
          S.species_names_for(tr) == tr)
    check("a species list passes through untouched", S.species_names_for(S.CWD12_SPECIES) == S.CWD12_SPECIES)
    check("species_to_legacy inverts legacy_to_species",
          all(S.species_to_legacy(S.legacy_to_species(n)) == n for n in S.CWD12_LEGACY_LABELS))
    check("the slot table agrees with mega_trainer", S.CWD12_ID_TO_SLOT == M.CWD12_ORIG_TO_CANON)
    check("binomials and common names cover all twelve",
          set(S.CWD12_BINOMIAL) == set(S.CWD12_SPECIES) == set(S.CWD12_COMMON))

    # class_species: copies by id, legacy lists by label, everything else by name
    stale_four = ["Eclipta", "Goosegrass", "Morningglory", "Nutsedge"]
    check("cottonweed_holdout id 0 is Waterhemp whatever its stored names say",
          S.class_species("cottonweed_holdout", 0, stale_four) == "Waterhemp")
    check("cottonweed_holdout id 11 resolves although the stored list has 4 names",
          S.class_species("cottonweed_holdout", 11, stale_four) == "CutleafGroundcherry")
    check("cottonweed_sp8 local id 2 is Eclipta (slot 2)",
          S.class_species("cottonweed_sp8", 2, S.TRAINER_SLOT_LEGACY[:8]) == "Eclipta")
    check("a real 'Ragweed' is ragweed", S.class_species("project_agml__weed_crop_detection", 0, ["Ragweed"]) == "Ragweed")
    check("'Ragweed' inside a whole legacy list is Sicklepod",
          S.class_species("some_export", 9, S.CWD12_LEGACY_LABELS) == "Sicklepod")
    check("a real 'Crabgrass' is no cwd12 species",
          S.class_species("rf_zig-zag-lnodr__weed-detection-vanpe", 1,
                          ["Carpet weed", "Crabgrass", "Eclipta", "Goosegrass", "Morning glory", "Nutsedge"]) is None)
    check("an uploaded 'Nutsedge' box in the cwd12 gold project is Ragweed",
          S.uploaded_label_species("cwd12-multiclass-v1", "Nutsedge") == "Ragweed")
    check("a 'Nutsedge' class in a real-name project is no cwd12 species",
          S.uploaded_label_species("weed-crop-agent-clean", "Nutsedge") is None)

    from weed_optimizer_framework.config import Config
    check("Config.ALL_CLASSES names the true species by cwd12 id",
          [Config.ALL_CLASSES[i] for i in range(12)] == S.CWD12_SPECIES)
    check("the sp8 registration list is the species of its local ids",
          [Config.ALL_CLASSES[i] for i in sorted(Config.TRAIN_SPECIES_IDS)] == S.CWD12_ID_SPACE["cottonweed_sp8"])
    check("the ImageWeeds cross-dataset test set never trains",
          "project_agml__imageweeds_weed_detection" in M.NEVER_TRAIN_SLUGS)
    got = M._build_canonical_class_map("rf_export_of_our_cwd12", {"class_names": S.CWD12_LEGACY_LABELS, "annotation": "yolo"})[0]
    check("a dataset carrying our whole legacy list is read as legacy labels, id by id",
          got == M.CWD12_ORIG_TO_CANON, got)

    # --- the three external datasets that were mis-joined ---------------------
    three = ["Carpetweed", "Eclipta", "Goosegrass", "Lambsquarters", "MorningGlory",
             "PalmerAmaranth", "Purslane", "Ragweed", "SpottedSpurge", "Waterhemp"]
    got = slot_species(M._build_canonical_class_map(
        "project_agml__three_season_weed_detection", {"class_names": three})[0])
    check("3SeasonWeedDet10: every species lands in its own slot",
          all(got[i] == n for i, n in enumerate(three) if n != "Lambsquarters"), got)
    check("3SeasonWeedDet10: Lambsquarters keeps its box, in an aux slot",
          got[three.index("Lambsquarters")] == "aux", got)

    zig = ["Carpet weed", "Crabgrass", "Eclipta", "Goosegrass", "Morning glory", "Nutsedge"]
    got = slot_species(M._build_canonical_class_map(
        "rf_zig-zag-lnodr__weed-detection-vanpe",
        {"class_names": zig, "annotation": "yolo"})[0])
    check("Roboflow set: Crabgrass and Nutsedge go to aux, not to dicot slots",
          got[1] == "aux" and got[5] == "aux", got)
    check("Roboflow set: 'Carpet weed' and 'Morning glory' are kept and joined",
          got[0] == "Carpetweed" and got[4] == "MorningGlory", got)

    agml = ["Blackbean", "Canola", "Corn", "Field Pea", "Flax", "Horseweed", "Kochia",
            "Lentil", "Palmer Amaranth", "Ragweed", "Redroot Pigweed", "Soybean",
            "Sugar beet", "Waterhemp"]
    got = slot_species(M._build_canonical_class_map(
        "project_agml__greenhouse_crop_weed_detection",
        {"class_names": agml, "annotation": "bbox"})[0])
    check("AgML greenhouse: Ragweed joins Ragweed, not Sicklepod", got[9] == "Ragweed", got)
    check("AgML greenhouse: crops stay in aux slots",
          all(got[agml.index(c)] == "aux" for c in ("Corn", "Soybean", "Canola")), got)

    # --- the two cwd12 copies -------------------------------------------------
    hold = M._build_canonical_class_map(
        "cottonweed_holdout",
        {"class_names": ["Eclipta", "Goosegrass", "Morningglory", "Nutsedge"]})[0]
    check("leave4out/dataset_holdout is read in original cwd12 ids, not by its 4 names",
          hold == M.CWD12_ORIG_TO_CANON, hold)
    sp8 = M._build_canonical_class_map(
        "cottonweed_sp8", {"class_names": M.CANONICAL_12_NAMES[:8]})[0]
    check("leave4out/dataset_8species is read in slot ids 0-7", sp8 == {i: i for i in range(8)})

    # --- the per-merge check catches a copy whose ids have drifted -----------
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        ref = tmp / "cwd12" / "train" / "labels"
        ref.mkdir(parents=True)
        copy = tmp / "copy" / "train" / "labels"
        copy.mkdir(parents=True)
        for k in range(12):
            box = "0.%02d 0.50 0.10 0.10" % (10 + 5 * k)
            (ref / ("img%02d.txt" % k)).write_text("%d %s\n" % (k, box))
            (copy / ("img%02d.txt" % k)).write_text("%d %s\n" % (k, box))
        real = M._cwd12_train_label_dir
        M._cwd12_train_label_dir = lambda: ref
        try:
            a, n = M._verify_cwd12_copy(tmp / "copy", dict(M.CWD12_ORIG_TO_CANON))
            check("a copy in original ids passes with the original-id map",
                  n == 12 and a == 12, (a, n))
            a, n = M._verify_cwd12_copy(tmp / "copy", M._species_class_map(
                "x", ["Eclipta", "Goosegrass", "Morningglory", "Nutsedge"]))
            check("the old four-name reading of that copy is caught",
                  n == 12 and a < 0.98 * n, (a, n))
        finally:
            M._cwd12_train_label_dir = real

    # --- near-duplicate guard -------------------------------------------------
    import random
    rng = random.Random(7)
    idx = M._NearHashIndex()
    base = rng.getrandbits(64)
    idx.add(base, M.HOLDOUT_HASH_SENTINEL)

    def flip(h, bits):
        for b in rng.sample(range(64), bits):
            h ^= 1 << b
        return h

    check("an exact copy of a holdout image is found at 0 bits",
          idx.find(base) == (M.HOLDOUT_HASH_SENTINEL, 0))
    for k in (1, 2, 3):
        got = idx.find(flip(base, k))
        check("a re-encoded holdout copy %d bit(s) away is caught" % k,
              got == (M.HOLDOUT_HASH_SENTINEL, k), got)
    for k in (4, 5, 6):
        got = idx.find(flip(base, k))
        check("a holdout copy %d bits away is caught (HOLDOUT_NEAR_DUP_BITS)" % k,
              got == (M.HOLDOUT_HASH_SENTINEL, k), got)
    check("a different photo 7 bits from a holdout image is not a copy",
          idx.find(flip(base, 7)) is None)
    other = M._NearHashIndex()
    other.add(base, ("rf_x", "a.jpg"))
    check("a non-holdout image keeps the 3-bit range",
          other.find(flip(base, 3)) is not None and other.find(flip(base, 4)) is None)
    # the two holdout re-exports the registry scan found (lab, 2026-09-21)
    idx = M._NearHashIndex()
    idx.add(0x76ad185cd8d86c27, (M.HOLDOUT_HASH_SENTINEL, None))  # test/20210820_iPhoneSE_YL_1480
    idx.add(0x10044e063418d8ba, (M.HOLDOUT_HASH_SENTINEL, None))  # test/20210806_iPhoneSE_YL_353
    for name, h, bits in (("rf_zig-zag Morningglory_891 re-export", 0x76ed185cd8da7c67, 4),
                          ("rf_karthikeya SpottedSpurge_86 re-export", 0x10044e063418d874, 5)):
        hits = idx.matches(h)
        check("%s (%d bits) is blocked as holdout" % (name, bits),
              M._dedup_verdict(hits, "rf_x") == ("holdout", bits), hits)

    stored = [rng.getrandbits(64) for _ in range(3000)]
    idx = M._NearHashIndex()
    for i, h in enumerate(stored):
        idx.add(h, i)
    queries = [flip(rng.choice(stored), rng.randint(0, 5)) for _ in range(400)]
    queries += [rng.getrandbits(64) for _ in range(100)]
    agree = 0
    for q in queries:
        brute = min(((bin(q ^ h).count("1"), i) for i, h in enumerate(stored)))
        want = brute[0] if brute[0] <= M.NEAR_DUP_BITS else None
        got = idx.find(q)
        agree += (got is None and want is None) or (got is not None and got[1] == want)
    check("the block index agrees with a brute-force scan on 500 queries",
          agree == len(queries), "%d/%d" % (agree, len(queries)))
    agree = 0
    for q in queries:
        brute = sorted(d for d in (bin(q ^ h).count("1") for h in stored) if d <= M.NEAR_DUP_BITS)
        agree += [b for _o, b in idx.matches(q)] == brute
    check("matches() returns every stored hash within range, nearest first",
          agree == len(queries), "%d/%d" % (agree, len(queries)))

    # --- which near matches block an image ------------------------------------
    H = (M.HOLDOUT_HASH_SENTINEL, None)
    V = M._dedup_verdict
    check("a holdout copy is blocked even when another dataset's copy is nearer",
          V([(("rf_x", "a.jpg"), 0), (H, 2)], "rf_y") == ("holdout", 2))
    check("an exact duplicate from any dataset is blocked",
          V([(("rf_x", "a.jpg"), 0)], "rf_y") == ("duplicate", 0))
    check("a near copy of a verified cwd12 photo is blocked",
          V([(("cottonweed_sp8", "b.jpg"), 1)], "rf_agrobot") == ("near_cwd12", 1))
    check("a near match to an ordinary dataset is NOT a duplicate (padded / studio images)",
          V([(("rf_srec", "c.jpg"), 2)], "rf_srec") is None)
    check("two different cwd12 photos a few bits apart are both kept",
          V([(("cottonweed_sp8", "d.jpg"), 3)], "cottonweed_holdout") is None)

    print("\n%d failure(s)" % len(FAILURES))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
