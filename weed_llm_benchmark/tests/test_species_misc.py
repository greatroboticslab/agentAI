#!/usr/bin/env python3
"""The auxiliary tools name cwd12 classes by species, never by the legacy labels.

Covers the bucket audit, the topic tagger, the OWLv2 sample-audit prompts, the
cut-paste object bank (legacy folder names read as species, new banks keyed by
species), DINOv2 routing, the DINO label verifier's slug map, the FLUX prompts,
per-class AP naming, the round manifest, the sample gallery, the harvest
search terms, the holdout guard of the object bank, the exemplar -> YOLO
export, the active-learning round, the FLUX LoRA class argument and the Mongo
class seed.

Run:  python3 tests/test_species_misc.py
"""
import os
import pathlib
import shutil
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="species_misc_")
os.environ["REPO_ROOT"] = TMP      # modules that mkdir under REPO at import
os.environ["CLASS_TOPIC_OVERRIDES_FILE"] = os.path.join(TMP, "class_topic_overrides.json")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from weed_optimizer_framework.tools import cwd12_species as S  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def _write(p, text=""):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def test_bucketer():
    print("bucketer")
    from weed_optimizer_framework.tools import bucketer as B
    check("coverage is reported over the twelve species", list(B.CWD12) == S.CWD12_SPECIES)
    root = pathlib.Path(TMP) / "bk"
    # holdout files use the original twelve ids; its stored names are four
    _write(root / "h" / "labels" / "a.txt", "0 .5 .5 .1 .1\n5 .5 .5 .1 .1\n")
    c = B._scan_bucket_a_species(root / "h", [root / "h" / "labels"],
                                 ["Eclipta", "Goosegrass", "Morningglory", "Nutsedge"],
                                 slug="cottonweed_holdout")
    check("cottonweed_holdout ids resolve by id space", dict(c) == {"Waterhemp": 1, "Ragweed": 1}, dict(c))
    _write(root / "r" / "labels" / "a.txt", "0 .5 .5 .1 .1\n1 .5 .5 .1 .1\n2 .5 .5 .1 .1\n")
    c = B._scan_bucket_a_species(root / "r", [root / "r" / "labels"],
                                 ["Ragweed", "Crabgrass", "Carpet weed"], slug="rf_x")
    check("a real Ragweed counts as ragweed, Crabgrass not at all",
          dict(c) == {"Ragweed": 1, "Carpetweed": 1}, dict(c))


def test_topic():
    print("topic_classifier")
    from weed_optimizer_framework.tools import topic_classifier as T
    check("Crabgrass is not a cwd12 topic", T.classify_keyword("Crabgrass")[0] == "weed")
    check("Nutsedge is not a cwd12 topic", T.classify_keyword("Nutsedge")[0] == "weed")
    check("Waterhemp is a cwd12 topic", T.classify_keyword("Waterhemp")[0] == "cwd12")
    check("Cutleaf groundcherry is a cwd12 topic",
          T.classify("Cutleaf groundcherry", use_llm=False, persist=False)["topic"] == "cwd12")
    check("Giant ragweed is not cwd12", T.classify_keyword("Giant ragweed")[0] != "cwd12")


def test_sample_audit():
    print("sample_audit")
    from weed_optimizer_framework.tools.sample_audit import _prompt_class_names as P
    h = P("cottonweed_holdout", {"class_names": ["Eclipta", "Goosegrass", "Morningglory", "Nutsedge"]})
    check("holdout prompts name all twelve species by id", h[0] == "Waterhemp" and len(h) == 12, h)
    sp8 = P("cottonweed_sp8", {"class_names": list(S.TRAINER_SLOT_LEGACY[:8])})
    check("sp8 prompts follow its local ids", sp8[1] == "Morning glory" and "Crabgrass" not in sp8, sp8)
    check("a real-name slug is prompted as stored",
          P("rf_zig-zag", {"class_names": ["Crabgrass"]}) == ["Crabgrass"])


def test_bank():
    print("synth_cutpaste bank")
    from weed_optimizer_framework.tools import synth_cutpaste as C
    check("CANONICAL_12 is the species in cwd12 id order", C.CANONICAL_12 == S.CWD12_SPECIES)
    check("holdout cid 0 is banked as Waterhemp",
          C._canon_name_for_label("cottonweed_holdout", {"class_names": ["Eclipta"]}, 0) == "Waterhemp")
    check("an external real Ragweed is banked as Ragweed",
          C._canon_name_for_label("agml_x", {"class_names": ["Ragweed"]}, 0) == "Ragweed")
    check("an external Crabgrass is dropped",
          C._canon_name_for_label("rf_zig-zag", {"class_names": ["Crabgrass"]}, 0) is None)
    legacy = pathlib.Path(TMP) / "bank_legacy"
    for d in ("Crabgrass", "Ragweed", "not_weed"):
        _write(legacy / d / "cottonweed_sp8_a_0000.png")
    _write(legacy / "Crabgrass" / "cottonweed_holdout_b_0000.png")
    dirs = {sp: d.name for sp, d in C.bank_class_dirs(legacy)}
    check("legacy bank folders read as species",
          dirs == {"MorningGlory": "Crabgrass", "Sicklepod": "Ragweed"}, dirs)
    check("a legacy bank's holdout crops are not used",
          not C.bank_crop_usable(legacy, legacy / "Crabgrass" / "cottonweed_holdout_b_0000.png"))
    check("a NEVER_TRAIN crop is not used in either vocabulary",
          not C.bank_crop_usable(legacy, legacy / "Crabgrass" / "cottonweeddet12_b_0000.png"))
    sp = pathlib.Path(TMP) / "bank_species"
    _write(sp / C.BANK_VOCAB_FILE, "species\n")
    _write(sp / "Ragweed" / "x.png")
    check("a species bank's Ragweed folder is ragweed",
          [s for s, _ in C.bank_class_dirs(sp)] == ["Ragweed"])
    check("new banks are written to their own directory",
          C.SPECIES_BANK_DIR != C.BANK_DIR and C.default_bank_dir() == C.BANK_DIR)

    print("dinov2_route")
    from weed_optimizer_framework.tools import dinov2_route as R
    bank = R.build_exemplar_bank(legacy)
    labels = sorted({s for s, _ in bank})
    check("routing labels are species (not_weed kept)",
          labels == ["MorningGlory", "Sicklepod", "not_weed"], labels)
    check("routing never offers a holdout crop",
          all(not p.name.startswith("cottonweed_holdout_") for _, p in bank))
    check("is_cwd12_match set is the species", set(R.CWD12) == set(S.CWD12_SPECIES))


def test_verifier_and_prompts():
    print("dino_label_verifier / synth_diffusion")
    from weed_optimizer_framework.tools import dino_label_verifier as V
    m = V._slug_canon_map("cottonweed_holdout", {"class_names": ["Eclipta", "Goosegrass"]})
    check("holdout map covers twelve ids by species", m and m[1] == "MorningGlory" and len(m) == 12)
    m = V._slug_canon_map("rf_zig-zag", {"class_names": ["Crabgrass", "Nutsedge", "Ragweed"]})
    check("real names map by species only", m == {2: "Ragweed"}, m)
    from weed_optimizer_framework.tools import synth_diffusion as D
    check("FLUX prompts are keyed by species", set(D.SPECIES_PROMPT) == set(S.CWD12_SPECIES))
    check("no prompt asks for crabgrass or nutsedge",
          not any(w in v for v in D.SPECIES_PROMPT.values() for w in ("crabgrass", "nutsedge")))
    check("FLUX output no longer mixes into the legacy-id directory",
          D.DIFF_DIR.name == "synth_diffusion_species")


def test_meta_and_manifest():
    print("build_result_meta / train_yolo_on_verified / dashboard_samples")
    from weed_optimizer_framework.tools import build_result_meta as M

    class G:
        dataset = {"categories": [{"id": i, "name": n} for i, n in enumerate(S.TRAINER_SLOT_LEGACY)]}
    check("a legacy GT category list is named by species",
          M._category_species(G()) == S.TRAINER_SLOT_SPECIES)
    G.dataset = {"categories": [{"id": i, "name": str(i)} for i in range(12)]}
    check("unnamed categories fall back to trainer slot species",
          M._category_species(G()) == S.TRAINER_SLOT_SPECIES)

    import json
    fw = pathlib.Path(TMP) / "results" / "framework"
    _write(fw / "dataset_registry.json", json.dumps({"datasets": {
        "cottonweed_sp8": {"status": "downloaded", "local_path": ""}}}))
    from weed_optimizer_framework.tools import train_yolo_on_verified as T
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        r = T.run(0, dry_run=True)
    prev = r.get("manifest_preview", "")
    check("round manifest names cwd12 ids by species",
          "0: Waterhemp" in prev and "Crabgrass" not in prev, prev[:200])

    try:
        from weed_optimizer_framework.tools import dashboard_samples  # noqa: F401
        src = pathlib.Path(dashboard_samples.__file__).read_text()
        check("sample gallery has no legacy class list", "\"Crabgrass\"" not in src)
    except ImportError as e:
        print("  skip dashboard_samples (not importable here: %s)" % e)


def test_harvest_queries():
    print("dataset_discovery harvest queries")
    from weed_optimizer_framework.tools.dataset_discovery import DatasetDiscovery as DD
    qs = DD.DEFAULT_HARVEST_QUERIES
    check("no crabgrass / nutsedge / cyperus queries",
          not [q for q in qs if any(w in q for w in ("crabgrass", "nutsedge", "cyperus"))])
    for q in ("waterhemp", "cutleaf groundcherry", "physalis angulata", "amaranthus tuberculatus"):
        check("query for %s" % q, q in qs)
    check("queries are unique", len(qs) == len(set(qs)))

    from weed_optimizer_framework.tools.web_identifier import WebIdentifier
    r = WebIdentifier()._identify_local("/data/waterhemp_001.jpg")
    check("web_identifier knows waterhemp", r["species"] == "Amaranthus tuberculatus")


def _img(p, seed, size=200):
    import numpy as np
    from PIL import Image
    p.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(p)


# the sealed cwd12 test split: one photograph, also copied into sp8 below
_HOLD = pathlib.Path(TMP) / "downloads" / "cottonweeddet12" / "test" / "images" / "hold1.jpg"
_img(_HOLD, 1)


def test_bank_holdout_guard():
    print("synth_cutpaste holdout guard")
    from weed_optimizer_framework.tools import synth_cutpaste as C
    dl = pathlib.Path(TMP) / "downloads"
    _img(dl / "cottonweed_sp8" / "images" / "a.jpg", 2)
    _write(dl / "cottonweed_sp8" / "labels" / "a.txt", "0 .5 .5 .6 .6\n")
    shutil.copy(_HOLD, dl / "cottonweed_sp8" / "images" / "hold1.jpg")
    _write(dl / "cottonweed_sp8" / "labels" / "hold1.txt", "0 .5 .5 .6 .6\n")
    shutil.copy(_HOLD, dl / "cottonweed_sp8" / "images" / "renamed.jpg")
    _write(dl / "cottonweed_sp8" / "labels" / "renamed.txt", "0 .5 .5 .6 .6\n")
    _img(dl / "cottonweeddet12" / "train" / "images" / "t.jpg", 3)
    _write(dl / "cottonweeddet12" / "train" / "labels" / "t.txt", "1 .5 .5 .6 .6\n")
    meta = C.build_bank(max_per_class=50)
    check("bank holds only the sp8 non-holdout photo (Waterhemp = sp8 id 0)",
          meta["per_class"] == {"Waterhemp": 1}, meta["per_class"])
    check("stem and renamed copies of a test photo are skipped",
          meta["skipped_holdout_photos_per_slug"].get("cottonweed_sp8") == 2,
          meta["skipped_holdout_photos_per_slug"])
    check("NEVER_TRAIN slugs are not cropped",
          "cottonweeddet12" in meta["never_train_skipped"])
    check("the species bank is now the default", C.default_bank_dir() == C.SPECIES_BANK_DIR)
    sp = C.SPECIES_BANK_DIR
    check("a species-bank crop from a train photo is usable",
          C.bank_crop_usable(sp, sp / "Waterhemp" / "cottonweed_holdout_x_0000.png"))
    check("a crop of a holdout photo is not usable in either vocabulary",
          not C.bank_crop_usable(sp, sp / "Waterhemp" / "cottonweed_sp8_hold1_0000.png")
          and not C.bank_crop_usable(C.BANK_DIR, C.BANK_DIR / "Crabgrass" / "cottonweed_sp8_hold1_0000.png"))


def test_active_round_and_lora():
    print("active_learning_round / flux_lora_train")
    import json
    from weed_optimizer_framework.tools import active_learning_round as A
    check("round species are the twelve species", list(A.CWD12) == S.CWD12_SPECIES)
    check("aliases are accepted", A.species_arg("Amaranthus tuberculatus") == "Waterhemp")
    check("Crabgrass / Nutsedge are refused",
          A.species_arg("Crabgrass") is None and A.species_arg("Nutsedge") is None)
    ex = A.gather_green_exemplars("Waterhemp")
    check("exemplars come from the species bank", len(ex) == 1 and "object_bank_species" in ex[0]["image"], ex)
    check("no MorningGlory exemplar in the bank", A.gather_green_exemplars("MorningGlory") == [])
    cfg = A.write_exemplar_config("Waterhemp", ex, pathlib.Path(TMP) / "cfg")
    d = json.loads(cfg.read_text())
    check("the OWL config declares its vocabulary", d.get("vocabulary") == "species" and d["species"] == "Waterhemp")
    check("new rounds do not share the legacy rounds directory",
          A.ROUND_DIR.name == "active_learning_rounds_species")

    from weed_optimizer_framework.tools import flux_lora_train as L
    check("LoRA classes are species", L.CANONICAL_12 == S.CWD12_SPECIES)
    check("LoRA class arg: Crabgrass refused, alias accepted",
          L._species_arg("Crabgrass") is None and L._species_arg("Palmer amaranth") == "PalmerAmaranth")
    crops = L._load_class_crops("Waterhemp", min_px=10)
    check("LoRA crops come from the species folder", len(crops) == 1 and crops[0].parent.name == "Waterhemp", crops)
    check("LoRA output does not reuse legacy-named folders", L.LORA_DIR.name == "flux_lora_species")


def test_exemplar_export():
    print("exemplar_to_yolo_dataset")
    import json
    from weed_optimizer_framework.tools import exemplar_to_yolo_dataset as E
    check("export classes are the species", E.CWD12 == S.CWD12_SPECIES)
    fw = pathlib.Path(TMP) / "results" / "framework"
    rf = pathlib.Path(TMP) / "rfx"
    _img(rf / "images" / "r.jpg", 4)
    _write(rf / "labels" / "r.txt", "0 .5 .5 .2 .2\n1 .4 .4 .2 .2\n")
    reg = {"datasets": {
        "rf_x": {"class_names": ["Ragweed", "Crabgrass"], "local_path": str(rf)},
        "cottonweeddet12": {"class_names": list(S.CWD12_LEGACY_LABELS), "local_path": str(rf)},
    }}
    # legacy bank folder Goosegrass = SpottedSpurge
    _write(E.BANK_DIR / "Goosegrass" / "cottonweed_sp8_q_0000.png")
    _write(E.BANK_DIR / "Crabgrass" / "cottonweed_sp8_m_0000.png")
    _write(E.BANK_DIR / "Crabgrass" / "cottonweed_holdout_m_0000.png")
    _write(fw / "synth_diffusion" / "images" / "f1.jpg")
    logs = fw / "class_exemplars"
    _write(logs / "SpottedSpurge.jsonl", "\n".join(json.dumps(e) for e in [
        {"img": "bank/Goosegrass/cottonweed_sp8_q_0000.png", "verdict": "exemplar", "ts": 1, "class": "SpottedSpurge"},
        {"img": "reg/rf_x/r.jpg", "verdict": "exemplar", "ts": 2, "class": "Ragweed"},
        {"img": "reg/cottonweeddet12/r.jpg", "verdict": "exemplar", "ts": 3, "class": "Waterhemp"},
    ]) + "\n")
    # a pre-v3.60.0 log named by a legacy label, events without "class"
    _write(logs / "Crabgrass.jsonl", "\n".join(json.dumps(e) for e in [
        {"img": "bank/Crabgrass/cottonweed_sp8_m_0000.png", "verdict": "exemplar", "ts": 4},
        {"img": "bank/Crabgrass/cottonweed_holdout_m_0000.png", "verdict": "exemplar", "ts": 5},
        {"img": "flux/f1.jpg", "verdict": "exemplar", "ts": 6},
    ]) + "\n")
    ex = E._read_all_exemplars(reg)
    got = {k: sorted(e["img"] for e in v) for k, v in ex.items()}
    check("events keyed by their class; a legacy Crabgrass log is MorningGlory",
          set(got) == {"SpottedSpurge", "Ragweed", "Waterhemp", "MorningGlory"}, got)
    guard = E._holdout_guard()
    r, _ = E._resolve_source("SpottedSpurge", "bank/Goosegrass/cottonweed_sp8_q_0000.png", reg, guard)
    check("SpottedSpurge bank crop gets cwd12 id 3", r and r[2] == ["3 0.5 0.5 0.95 0.95"], r)
    r, _ = E._resolve_source("Ragweed", "reg/rf_x/r.jpg", reg, guard)
    check("a real Ragweed box gets id 5, the Crabgrass box is dropped",
          r and r[2] == ["5 .5 .5 .2 .2"], r)
    r, why = E._resolve_source("Waterhemp", "reg/cottonweeddet12/r.jpg", reg, guard)
    check("NEVER_TRAIN reg exemplar is refused", r is None and why == "holdout", why)
    r, why = E._resolve_source("MorningGlory", "bank/Crabgrass/cottonweed_holdout_m_0000.png", reg, guard)
    check("legacy-bank holdout crop is refused", r is None and why == "holdout", why)
    r, why = E._resolve_source("MorningGlory", "flux/f1.jpg", reg, guard)
    check("legacy FLUX output is refused", r is None and why == "flux_legacy_ids", why)
    r, why = E._resolve_source("MorningGlory", "bank/Goosegrass/cottonweed_sp8_q_0000.png", reg, guard)
    check("a crop from another species' folder is refused", r is None and why == "bank_other_species", why)
    old = pathlib.Path(TMP) / "exy_old"
    _write(old / "data.yaml", "names: %s\n" % json.dumps(S.CWD12_LEGACY_LABELS))
    new = pathlib.Path(TMP) / "exy_new"
    _write(new / "data.yaml", "names: %s\n" % json.dumps(S.CWD12_SPECIES))
    check("a legacy output dir is refused", E._out_dir_conflict(old) is not None)
    check("a species output dir is accepted", E._out_dir_conflict(new) is None
          and E._out_dir_conflict(pathlib.Path(TMP) / "exy_none") is None)


def test_v3600_cross_package():
    """Contracts that span packages: bank / FLUX / background dirs that the
    producers and consumers must agree on, and the NEVER_TRAIN exclusions."""
    print("cross-package contracts")
    import json
    import re
    from weed_optimizer_framework.tools import synth_cutpaste as C
    from weed_optimizer_framework.tools import exemplar_to_yolo_dataset as E
    fw = pathlib.Path(TMP) / "results" / "framework"

    # backgrounds: the unguarded legacy dir is never used; a guarded run skips
    # the holdout photo by stem and its renamed copy by dHash
    _write(C.LEGACY_BG_DIR / "bg_0000.jpg")
    check("legacy backgrounds/ is not the background dir",
          C.BG_DIR != C.LEGACY_BG_DIR and C.guarded_backgrounds() == [])
    n = C.collect_backgrounds(n=10, size=32)
    check("guarded backgrounds: only the non-holdout sp8 photo", n == 1, n)
    check("guarded run is marked and used",
          (C.BG_DIR / C.BG_GUARD_FILE).is_file() and len(C.guarded_backgrounds()) == 1)

    # FLUX: species-era names never collide with legacy ones
    from weed_optimizer_framework.tools import synth_diffusion as SD
    check("FLUX species prefix agrees between producer and exporter",
          SD.FLUX_SPECIES_PREFIX == E.FLUX_SPECIES_PREFIX == "fluxsp_"
          and SD.DIFF_DIR.resolve() == E.FLUX_SPECIES_DIR.resolve())
    _write(fw / "synth_diffusion" / "images" / "fluxsynth_Ragweed_000000.jpg")
    _write(fw / "synth_diffusion_species" / "images" / "fluxsp_Ragweed_000000.jpg")
    _write(fw / "synth_diffusion_species" / "labels" / "fluxsp_Ragweed_000000.txt",
           "5 .5 .5 .2 .2\n")
    guard = E._holdout_guard()
    r, why = E._resolve_source("Ragweed", "flux/fluxsp_Ragweed_000000.jpg", {}, guard)
    check("a species-era FLUX image resolves beside a legacy one of the same class",
          r is not None and r[2] == ["5 .5 .5 .2 .2"], why)
    r, why = E._resolve_source("Ragweed", "flux/fluxsynth_Ragweed_000000.jpg", {}, guard)
    check("the legacy FLUX image is still refused", r is None and why == "flux_legacy_ids", why)

    # species bank crops are keyed 'banksp/', legacy ones 'bank/'
    crop = "cottonweed_sp8_a_0000.png"
    r, why = E._resolve_source("Waterhemp", "banksp/Waterhemp/" + crop, {}, guard)
    check("a 'banksp/' key reads the species bank", r is not None and r[0] == "bank", why)
    r, why = E._resolve_source("Waterhemp", "bank/Waterhemp/" + crop, {}, guard)
    check("a 'bank/' key never reads the species bank", r is None, why)

    # export_owl_exemplars --source classes reads the dashboard's jsonl logs
    rf = pathlib.Path(TMP) / "rfx"
    _write(fw / "dataset_registry.json", json.dumps({"datasets": {
        "rf_x": {"class_names": ["Ragweed", "Crabgrass"], "local_path": str(rf)}}}))
    from weed_optimizer_framework.tools import export_owl_exemplars as X
    got = X._classes_exemplars("SpottedSpurge", 5)
    check("--source classes reads a bank verdict from class_exemplars/*.jsonl",
          len(got) == 1 and got[0]["image"].endswith("/Goosegrass/cottonweed_sp8_q_0000.png")
          and got[0]["bbox_yolo"] == [0.5, 0.5, 1.0, 1.0], got)
    got = X._classes_exemplars("Ragweed", 5)
    check("--source classes: a reg verdict keeps only this species' box",
          len(got) == 1 and got[0]["bbox_yolo"] == [0.5, 0.5, 0.2, 0.2], got)

    # NEVER_TRAIN slugs stay off the Roboflow round trip and the train manifest
    from weed_optimizer_framework.tools import roboflow_sync as RS
    nu = RS._never_upload_slugs()
    check("NEVER_TRAIN non-cwd12 slugs are never uploaded",
          {"weedsense", "francesco__weed_crop_aerial",
           "project_agml__imageweeds_weed_detection"} <= nu and "cottonweeddet12" not in nu, nu)
    _write(fw / "dataset_registry.json", json.dumps({"datasets": {
        "francesco__weed_crop_aerial": {"status": "downloaded", "harvest_round": 1},
        "rf_ok": {"status": "downloaded", "harvest_round": 1}}}))
    _write(fw / "slug_verdicts.jsonl", "\n".join(json.dumps({"slug": s, "verdict": "keep"})
                                               for s in ("francesco__weed_crop_aerial", "rf_ok")))
    from weed_optimizer_framework.tools import train_yolo_on_verified as TV
    v, _ = TV._verified_slugs(1)
    check("a kept NEVER_TRAIN slug is not in the train manifest", v == {"rf_ok"}, v)

    # the integrity audit joins real names through the species
    _write(fw / "dataset_registry.json", json.dumps({"datasets": {
        "rf_x": {"class_names": ["Ragweed", "Crabgrass"], "local_path": str(rf)}}}))
    from weed_optimizer_framework.tools import dataset_integrity_audit as IA
    a = IA.audit()
    check("integrity audit: real Ragweed counts as Ragweed, Crabgrass as nothing",
          a["per_class_images"] == {"Ragweed": 1} and len(a["species_missing"]) == 11, a["per_class_images"])

    # the campaign scripts pass species
    root = pathlib.Path(__file__).resolve().parents[1]
    arr = re.search(r"CANONICAL_12=\(([^)]*)\)",
                    (root / "run_v3_0_41_flux_12class_baseline.sh").read_text()).group(1).replace("\\", " ").split()
    check("FLUX 12-class array is the species in cwd12 id order", arr == S.CWD12_SPECIES, arr)
    arr = re.search(r"CLASSES=\(([^)]*)\)",
                    (root / "run_v3_0_41_lora_4class.sh").read_text()).group(1).split()
    check("LoRA 4-class array names species", all(S.cli_species(c) == c for c in arr), arr)


def test_backfill_and_topic_overrides():
    print("backfill_mongo / topic overrides")
    import json
    fw = pathlib.Path(TMP) / "results" / "framework"
    _write(pathlib.Path(os.environ["CLASS_TOPIC_OVERRIDES_FILE"]),
           json.dumps({"Crabgrass": "cwd12", "Lambsquarters": "weed", "Ragweed": "weed"}))
    _write(fw / "dataset_registry.json", json.dumps({"datasets": {
        "cottonweed_holdout": {"class_names": ["Eclipta", "Goosegrass", "Morningglory", "Nutsedge"]},
        "rf_zig-zag": {"class_names": ["Crabgrass", "Nutsedge", "Carpet weed"]},
        "legacy_copy": {"class_names": list(S.CWD12_LEGACY_LABELS)},
    }}))
    from weed_optimizer_framework.tools import backfill_mongo as B
    check("holdout classes are the twelve species by id",
          B._slug_classes("cottonweed_holdout", {"class_names": ["Eclipta"]}) == S.CWD12_SPECIES)
    check("real names map through species_of",
          B._slug_classes("rf_zig-zag", {"class_names": ["Crabgrass", "Carpet weed"]}) == ["Crabgrass", "Carpetweed"])

    class Coll:
        def __init__(self):
            self.ops = {}

        def update_one(self, flt, upd, upsert=False):
            self.ops[flt["_id"]] = upd

        def insert_one(self, doc):
            pass

        def estimated_document_count(self):
            return len(self.ops)

    class FakeDB(dict):
        def __missing__(self, k):
            self[k] = Coll()
            return self[k]

    from weed_optimizer_framework.tools import db as D
    fake = FakeDB()
    orig = D._get_db
    D._get_db = lambda: fake
    try:
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            B.apply()
    finally:
        D._get_db = orig
    cls = fake[D.COLL_CLASSES].ops
    wh = cls.get("Waterhemp", {}).get("$set", {})
    check("Waterhemp seeded as cwd12 id 0 with an English display name",
          wh.get("is_cwd12") and wh.get("cwd12_index") == 0 and wh.get("display_name") == "Waterhemp", wh)
    check("CutleafGroundcherry seeded as id 11",
          cls.get("CutleafGroundcherry", {}).get("$set", {}).get("cwd12_index") == 11)
    cg = cls.get("Crabgrass", {})
    check("Crabgrass is not cwd12, loses any cwd12 index, stale cwd12 tag ignored",
          cg.get("$set", {}).get("is_cwd12") is False and "cwd12_index" in cg.get("$unset", {})
          and cg.get("$set", {}).get("topic") != "cwd12", cg)
    check("no legacy-only name is seeded", not {"Carpetweeds", "Morningglory"} & set(cls))
    check("a species keeps topic cwd12 over a 'weed' override",
          cls["Ragweed"]["$set"]["topic"] == "cwd12")
    check("no Chinese in the seed", all(ord(ch) < 0x3000 for d in cls.values()
                                        for v in d.get("$set", {}).values() if isinstance(v, str) for ch in v))

    from weed_optimizer_framework.tools import topic_classifier as T
    r = T.classify("Crabgrass", use_llm=False, persist=False)
    check("a stale 'cwd12' override on Crabgrass is not honoured", r["topic"] == "weed", r)
    r = T.classify("Lambsquarters", use_llm=False, persist=False)
    check("other overrides still apply", r["topic"] == "weed" and r["source"] == "override", r)
    rb = T.classify_batch(["Crabgrass", "Ragweed"], use_llm=False, persist=False)
    check("classify_batch agrees", [x["topic"] for x in rb] == ["weed", "cwd12"], rb)
    check("topic_classifier has no unused legacy set", not hasattr(T, "_CWD12"))


def main():
    test_bucketer()
    test_topic()
    test_sample_audit()
    test_bank()
    test_bank_holdout_guard()
    test_active_round_and_lora()
    test_exemplar_export()
    test_v3600_cross_package()
    test_verifier_and_prompts()
    test_meta_and_manifest()
    test_harvest_queries()
    test_backfill_and_topic_overrides()
    print("\n%d failure(s)" % len(FAILURES))
    shutil.rmtree(TMP, ignore_errors=True)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
