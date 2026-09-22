#!/usr/bin/env python3
"""The dashboard names cwd12 classes by species, never by the legacy labels.

/classes used to fold every dataset's class name onto the alphabetical legacy
list, so one page mixed cwd12 boxes of one species with other datasets' boxes
of another (the "Ragweed" page held cwd12 Sicklepod beside real ragweed), and
cottonweed_holdout's four stored names were applied to ids 0-3 of its
twelve-id files. These pin the replacement on a small fake repository:

  * the class index keys cwd12 copies by id space and other slugs by real name;
  * a class pool selects boxes by the id that species has in each slug;
  * exemplar logs written before v3.60.0 are re-keyed on read, not rewritten;
  * Roboflow legacy labels are shown as "species (Roboflow label: X)";
  * legacy-only class URLs redirect, real Crabgrass/Nutsedge keep their page;
  * the annotation guide, the dataset analysis, the sample thumbnails and the
    static 12-class table all speak species.

Run:  python3 tests/test_species_dash.py
"""
import json
import os
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TMP = pathlib.Path(tempfile.mkdtemp(prefix="species_dash_"))
os.environ["REPO_ROOT"] = str(TMP)
(TMP / "dashpass").write_text("test-pass\n")
os.environ["DASHPASS_FILE"] = str(TMP / "dashpass")
os.environ["DASH_USER"] = "tester"
os.environ["ROBOFLOW_KEY_FILE"] = str(TMP / "no_roboflow_key")
os.environ["REG_INDEX_TTL_SEC"] = "0"

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


# ---------------------------------------------------------------- fake repo
def _write_split(root, files):
    """files: {stem: [class ids]} -> root/train/{images,labels}."""
    (root / "train" / "images").mkdir(parents=True, exist_ok=True)
    (root / "train" / "labels").mkdir(parents=True, exist_ok=True)
    for stem, ids in files.items():
        (root / "train" / "images" / f"{stem}.jpg").write_bytes(b"\xff\xd8fake")
        (root / "train" / "labels" / f"{stem}.txt").write_text(
            "".join(f"{i} 0.5 0.5 0.2 0.2\n" for i in ids))


LEGACY8 = ["Carpetweeds", "Crabgrass", "PalmerAmaranth", "PricklySida",
           "Purslane", "Ragweed", "Sicklepod", "SpottedSpurge"]
SP8 = TMP / "ds" / "sp8"
HOLD = TMP / "ds" / "holdout"
ZIG = TMP / "ds" / "zig"
AIML = TMP / "ds" / "aiml"
AGML = TMP / "ds" / "agml"
_write_split(SP8, {"sp8_a": [0], "sp8_b": [5], "sp8_c": [1, 2]})
_write_split(HOLD, {"hold_a": [0], "hold_b": [5], "hold_c": [2, 3]})
_write_split(ZIG, {"zig_a": [0], "zig_b": [1], "zig_c": [5]})
_write_split(AIML, {"aiml_a": [0]})
_write_split(AGML, {"agml_a": [0], "agml_b": [1]})

REG = {"datasets": {
    "cottonweed_sp8": {"class_names": LEGACY8, "local_path": str(SP8),
                       "status": "downloaded"},
    # the stored four names over twelve-id files, as the lab registry holds it
    "cottonweed_holdout": {"class_names": ["Eclipta", "Goosegrass", "Morningglory", "Nutsedge"],
                           "local_path": str(HOLD), "status": "downloaded"},
    "rf_zig": {"class_names": ["Carpet weed", "Crabgrass", "Eclipta", "Goosegrass",
                               "Morning glory", "Nutsedge"],
               "local_path": str(ZIG), "status": "downloaded"},
    "rf_aiml": {"class_names": ["Crab Grass"], "local_path": str(AIML),
                "status": "downloaded"},
    "agml_three": {"class_names": ["Ragweed", "Carpetweed"], "local_path": str(AGML),
                   "status": "downloaded"},
}}
(TMP / "results" / "framework").mkdir(parents=True, exist_ok=True)
(TMP / "results" / "framework" / "dataset_registry.json").write_text(json.dumps(REG))

from weed_optimizer_framework.tools import db as _db  # noqa: E402
_db.get_registry = lambda domain=None: json.loads(json.dumps(REG))

from weed_optimizer_framework.tools import dashboard_server as D  # noqa: E402
from weed_optimizer_framework.tools import cwd12_species as S  # noqa: E402

print("class index")
idx = D._load_registry_index()
sp8_space = S.CWD12_ID_SPACE["cottonweed_sp8"]
check("sp8 id 5 is Sicklepod, not the legacy 'Ragweed'",
      ("cottonweed_sp8", 5, "Sicklepod") in idx.get("Sicklepod", []), idx.get("Sicklepod"))
check("Ragweed = holdout id 5 + real ragweed, no sp8 boxes",
      sorted((s, c) for s, c, _ in idx.get("Ragweed", []))
      == [("agml_three", 0), ("cottonweed_holdout", 5)], idx.get("Ragweed"))
check("holdout indexed on all twelve ids despite four stored names",
      sorted(c for s, c, _ in sum(idx.values(), []) if s == "cottonweed_holdout")
      == list(range(12)))
check("Waterhemp = sp8 id 0 + holdout id 0",
      sorted((s, c) for s, c, _ in idx.get("Waterhemp", []))
      == [("cottonweed_holdout", 0), ("cottonweed_sp8", 0)])
check("real 'Carpet weed' and 'Carpetweed' join Carpetweed, not Waterhemp",
      sorted((s, c) for s, c, _ in idx.get("Carpetweed", []))
      == [("agml_three", 1), ("cottonweed_holdout", 4), ("rf_zig", 0)], idx.get("Carpetweed"))
check("real crabgrass spellings share a non-cwd12 class",
      sorted((s, c) for s, c, _ in idx.get("Crabgrass", []))
      == [("rf_aiml", 0), ("rf_zig", 1)], idx.get("Crabgrass"))
check("no class is keyed by a legacy-only label",
      not any(k in idx for k in ("Carpetweeds", "Morningglory")))
check("Nutsedge is real zig-zag nutsedge only",
      [(s, c) for s, c, _ in idx.get("Nutsedge", [])] == [("rf_zig", 5)])
check("Morning glory joins MorningGlory with sp8 id 1 and holdout id 1",
      sorted((s, c) for s, c, _ in idx.get("MorningGlory", []))
      == [("cottonweed_holdout", 1), ("cottonweed_sp8", 1), ("rf_zig", 4)])
check("topic: species are cwd12, crabgrass is a weed",
      D._class_topic("Waterhemp") == "cwd12" and D._class_topic("Crabgrass") == "weed")

print("class pools")
pool = D._reg_pool_for_class("Ragweed")
check("Ragweed pool: holdout photo with an id-5 box and the agml photo",
      sorted((e["slug"], e["fname"], e["cid"]) for e in pool)
      == [("agml_three", "agml_a.jpg", 0), ("cottonweed_holdout", "hold_b.jpg", 5)], pool)
pool = D._reg_pool_for_class("Eclipta")
check("Eclipta pool: the sp8 photo with a local id-2 box only",
      sorted((e["slug"], e["fname"], e["cid"]) for e in pool)
      == [("cottonweed_sp8", "sp8_c.jpg", 2)], pool)
cache = json.loads((D._pool_cache_dir / "Ragweed.json").read_text())
check("pool cache records its key space", cache.get("key_space") == "species")
(D._pool_cache_dir / "Sicklepod.json").write_text(json.dumps(
    {"reg_mtime": cache["reg_mtime"], "cap": 200,
     "entries": [{"kind": "reg", "slug": "x", "fname": "stale.jpg", "cid": 9}]}))
check("a pre-v3.60.0 pool cache is not served",
      all(e["fname"] != "stale.jpg" for e in D._reg_pool_for_class("Sicklepod")))

print("exemplar logs")
ex = D._CLS_EXEMPLAR_DIR


def _log(name, rows):
    with open(ex / f"{name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


_log("Carpetweeds", [
    {"img": "reg/cottonweed_sp8/sp8_a.jpg", "verdict": "exemplar", "ts": 10},
    {"img": "reg/rf_zig/zig_a.jpg", "verdict": "exemplar", "ts": 11},
    {"img": "cottonweed_sp8/Carpetweeds/sp8_a.jpg", "verdict": "bad", "ts": 12},
])
_log("Crabgrass", [
    {"img": "reg/cottonweed_sp8/sp8_c.jpg", "verdict": "exemplar", "ts": 20},
    {"img": "reg/rf_aiml/aiml_a.jpg", "verdict": "exemplar", "ts": 21},
    {"img": "reg/rf_zig/zig_b.jpg", "verdict": "bad", "ts": 22},
])
_log("Eclipta", [
    {"img": "reg/cottonweed_holdout/hold_a.jpg", "verdict": "exemplar", "ts": 30},
    {"img": "bank/Eclipta/crop1.png", "verdict": "exemplar", "ts": 31},
])
before = {p.name: p.read_bytes() for p in ex.glob("*.jsonl")}
check("sp8 entries under 'Carpetweeds' count for Waterhemp",
      D._exemplar_state("Waterhemp").get("reg/cottonweed_sp8/sp8_a.jpg") == "exemplar")
check("holdout entry under 'Eclipta' counts for Waterhemp (stored id 0)",
      D._exemplar_state("Waterhemp").get("reg/cottonweed_holdout/hold_a.jpg") == "exemplar")
check("zig-zag 'Carpet weed' entries count for Carpetweed",
      D._exemplar_state("Carpetweed") == {"reg/rf_zig/zig_a.jpg": "exemplar"},
      D._exemplar_state("Carpetweed"))
check("sp8 entries under 'Crabgrass' count for MorningGlory",
      D._exemplar_state("MorningGlory") == {"reg/cottonweed_sp8/sp8_c.jpg": "exemplar"})
check("real crabgrass entries stay Crabgrass",
      D._exemplar_state("Crabgrass") == {"reg/rf_aiml/aiml_a.jpg": "exemplar",
                                          "reg/rf_zig/zig_b.jpg": "bad"})
check("bank folder 'Eclipta' counts for its cwd12 id (Purslane)",
      D._exemplar_state("Purslane") == {"bank/Eclipta/crop1.png": "exemplar"})
check("no class keyed by a legacy-only label",
      "Carpetweeds" not in D._exemplar_index())
D._exemplar_append("Waterhemp", [{"img": "reg/cottonweed_sp8/sp8_a.jpg",
                                  "verdict": "bad", "ts": 99}])
check("a new verdict records its class and wins by time",
      D._exemplar_state("Waterhemp").get("reg/cottonweed_sp8/sp8_a.jpg") == "bad")
new_line = json.loads((ex / "Waterhemp.jsonl").read_text().splitlines()[-1])
check("new events carry 'class'", new_line.get("class") == "Waterhemp")
check("old logs are not rewritten",
      all((ex / n).read_bytes() == b for n, b in before.items()))

print("HTTP routes")
from fastapi.testclient import TestClient  # noqa: E402
import base64  # noqa: E402
client = TestClient(D.app)
client.headers["Authorization"] = "Basic " + base64.b64encode(b"tester:test-pass").decode()
r = client.get("/classes/Carpetweeds", follow_redirects=False)
check("/classes/Carpetweeds redirects to Waterhemp",
      r.status_code in (302, 307) and r.headers.get("location") == "/classes/Waterhemp",
      (r.status_code, r.headers.get("location")))
r = client.get("/classes/Morningglory", follow_redirects=False)
check("/classes/Morningglory redirects to Carpetweed",
      r.headers.get("location") == "/classes/Carpetweed")
r = client.get("/classes/Nutsedge")
check("/classes/Nutsedge renders the real nutsedge class",
      r.status_code == 200 and "not a cwd12 species" in r.text, r.status_code)
r = client.get("/classes/Ragweed")
check("/classes/Ragweed renders the species page",
      r.status_code == 200 and "Ambrosia artemisiifolia" in r.text, r.status_code)
r = client.get("/classes/NoSuchWeed")
check("/classes/unknown is a 404, not a 500", r.status_code == 404, r.status_code)
r = client.get("/audit/class/Nutsedge", follow_redirects=False)
check("/audit/class/Nutsedge redirects to Ragweed",
      r.headers.get("location") == "/audit/class/Ragweed")
r = client.get("/audit/class/Ragweed")
check("/audit/class/Ragweed shows species and legacy label",
      r.status_code == 200 and "legacy label <code>Nutsedge</code>" in r.text)
r = client.get("/audit")
check("/audit lists species with binomials",
      r.status_code == 200 and "Amaranthus tuberculatus" in r.text
      and "Digitaria" not in r.text and "Cyperus" not in r.text)
r = client.get("/audit/method")
check("/audit/method says the prompts mismatch the labels", "mislabelled by construction" in r.text)
r = client.get("/classes")
check("/classes landing lists species, not legacy-only labels",
      r.status_code == 200 and 'href="/classes/Waterhemp"' in r.text
      and 'href="/classes/Carpetweeds"' not in r.text)
exp = client.get("/api/exemplars_export").json()
wh = exp["by_class"].get("MorningGlory", {})
check("export: MorningGlory entry carries species and the log it came from",
      wh.get("species") == "MorningGlory"
      and wh["entries"][0].get("logged_as") == "Crabgrass"
      and wh["entries"][0].get("class_id_in_slug") == 1, wh)
check("export: bank entry points at its legacy folder",
      exp["by_class"]["Purslane"]["entries"][0]["thumb_url"].startswith("/thumb/bank/Eclipta/"))
one = client.get("/api/exemplars_export/Crabgrass").json()
check("export: Crabgrass is not a species", one.get("species") is None)
r = client.post("/api/exemplar/Carpetweeds", json={"img": "reg/cottonweed_sp8/sp8_a.jpg",
                                                  "verdict": "exemplar"})
check("a verdict under a legacy-only label is refused, not logged",
      r.status_code == 409 and ex.joinpath("Carpetweeds.jsonl").read_bytes()
      == before["Carpetweeds.jsonl"], r.status_code)
r = client.post("/api/exemplar/Nutsedge", json={"img": "reg/rf_zig/zig_c.jpg",
                                               "verdict": "exemplar"})
check("a verdict on the real Nutsedge class is logged with its class",
      r.status_code == 200
      and D._exemplar_state("Nutsedge") == {"reg/rf_zig/zig_c.jpg": "exemplar"})
ps = client.get("/api/per_species_stats").json()
check("per_species_stats keyed by species",
      set(ps["per_species"]) == set(S.CWD12_SPECIES)
      and [r["key"] for r in ps["species"]] == S.CWD12_SPECIES)

print("roboflow labels")
lab = D._rf_class_labels("cwd12-multiclass-v1", {"Ragweed": 5, "Nutsedge": 88})
check("legacy project: species first, Roboflow label second",
      {c["name"]: c["display"] for c in lab}
      == {"Ragweed": "Sicklepod on cwd12 photographs, Ragweed on others "
                     "(Roboflow label: Ragweed)",
          "Nutsedge": "Ragweed (Roboflow label: Nutsedge)"}, lab)
# v3.60.0: a legacy class also holds boxes on non-cwd12 photographs, named by
# their real reading (roboflow_sync._real_class_name); a non-legacy class
# exists there only for those.
lab = D._rf_class_labels("weed-crop-agent-dataset",
                         {"Carpetweeds": 4, "Waterhemp": 2, "Crabgrass (non-cwd12)": 1})
got = {c["name"]: (c["species"], c["species_other_photos"]) for c in lab}
check("legacy project: both readings of a legacy class, real name otherwise",
      got == {"Carpetweeds": ("Waterhemp", "Carpetweed"), "Waterhemp": ("Waterhemp", None),
              "Crabgrass (non-cwd12)": (None, None)}, got)
lab = D._rf_class_labels("weed-crop-agent-v2", {"Ragweed": 3, "Crabgrass": 2})
check("other project: real names as they are",
      {c["name"]: (c["species"], c["display"]) for c in lab}
      == {"Ragweed": ("Ragweed", "Ragweed"), "Crabgrass": (None, "Crabgrass")}, lab)

print("annotation guidance")
c = D._classify_dataset_classes(REG["datasets"]["rf_zig"]["class_names"])
check("vanpe-style list with real Crabgrass/Nutsedge is mixed, not cwd12",
      c["type"] == "mixed" and "Crabgrass" not in c["cwd12"] and "Nutsedge" not in c["cwd12"]
      and set(c["cwd12"]) == {"Carpetweed", "Eclipta", "Goosegrass", "MorningGlory"}, c)
c = D._classify_dataset_classes(["Cyperus rotundus", "Ipomoea"])
check("Cyperus is not cwd12; Ipomoea is MorningGlory",
      c["type"] == "mixed" and c["cwd12"] == ["MorningGlory"], c)
c = D._classify_dataset_classes(["Waterhemp", "Cutleaf groundcherry"])
check("Waterhemp and Cutleaf groundcherry are cwd12", c["type"] == "cwd12", c)
c = D._classify_dataset_classes(["Eclipta", "Goosegrass", "Morningglory", "Nutsedge"],
                                "cottonweed_holdout")
check("a cwd12 copy is classified by its id space",
      c["type"] == "cwd12" and len(c["cwd12"]) == 12, c)

print("dataset analysis")
old = {"ok": True, "annotations": {"type": "yolo", "classes": ["class 5", "class 0"],
                                   "per_class": {"class 5": 3, "class 0": 1}}}
tr = D._species_analysis("cottonweed_holdout", old)
check("cached holdout 'class N' keys become species",
      tr["annotations"]["per_class"] == {"Ragweed": 3, "Waterhemp": 1})
old = {"ok": True, "annotations": {"type": "yolo", "classes": ["Carpetweeds", "Ragweed"],
                                   "per_class": {"Carpetweeds": 1474, "Ragweed": 9}}}
tr = D._species_analysis("cottonweed_sp8", old)
check("cached sp8 legacy keys become species",
      tr["annotations"]["per_class"] == {"Waterhemp": 1474, "Sicklepod": 9})
check("a real-name slug's cache is left alone",
      D._species_analysis("agml_three", old) is old)
D._DATASET_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
D._resolve_slug_dir = lambda slug: {"cottonweed_sp8": SP8}.get(slug)
(SP8 / "data.yaml").write_text("names: " + json.dumps(LEGACY8) + "\n")
fresh = D._analyze_dataset("cottonweed_sp8", refresh=True)
check("fresh sp8 analysis names classes by species",
      fresh.get("annotations", {}).get("per_class")
      == {"Waterhemp": 1, "Sicklepod": 1, "MorningGlory": 1, "Eclipta": 1},
      fresh.get("annotations"))

print("sample thumbnails")
seen = {}


def _fake_render(img_path, label_path, out, slug, max_width=600, class_names=()):
    seen[slug] = list(class_names)
    return False


D.render_with_bbox = _fake_render
D.find_image_in_slug = lambda slug, fn: (SP8 / "train" / "images" / fn, SP8)
D.find_label_for_image = lambda img, root: None
client.get("/api/sample/cottonweed_sp8/sp8_a.jpg")
check("sp8 thumbnails label local ids by slot species",
      seen.get("cottonweed_sp8") == [S.CWD12_COMMON[s] for s in sp8_space], seen)
client.get("/api/sample/cottonweed_holdout/sp8_a.jpg")
check("holdout thumbnails label the twelve original ids",
      seen.get("cottonweed_holdout") == [S.CWD12_COMMON[s] for s in S.CWD12_SPECIES])

print("label verdicts from the chat montage")
check("class map by id: sp8 by its id space, zig-zag by real names",
      D._label_class_map("cottonweed_sp8").get("5") == sp8_space[5]
      and D._label_class_map("rf_zig").get("1") == "Crabgrass"
      and D._label_class_map("rf_zig").get("0") == "Carpetweed",
      (D._label_class_map("cottonweed_sp8"), D._label_class_map("rf_zig")))
r = client.post("/api/dataset/label_verdict",
                json={"slug": "cottonweed_sp8", "file": "sp8_b", "verdict": "bad",
                      "cid": "5", "cls": "Ragweed"})
check("a bad verdict is filed under the species of its class id, not the client string",
      r.json().get("forwarded") is True
      and D._exemplar_state(sp8_space[5]).get("cottonweed_sp8/sp8_b") == "bad"
      and "cottonweed_sp8/sp8_b" not in D._exemplar_state("Ragweed"), r.json())
r = client.post("/api/dataset/label_verdict",
                json={"slug": "rf_zig", "file": "zig_b", "verdict": "bad", "cls": "Crabgrass"})
check("without a class id nothing is forwarded", r.json().get("forwarded") is False, r.json())

print("object bank folders")
bank = TMP / "results" / "framework" / "synth_cutpaste" / "object_bank"
for folder in ("Crabgrass", "Nutsedge"):
    (bank / folder).mkdir(parents=True, exist_ok=True)
    (bank / folder / f"cottonweeddet12_{folder}_0.png").write_bytes(b"\x89PNG")
check("legacy-only labels have no bank folder",
      D._bank_dir_name("Crabgrass") is None and D._bank_dir_name("Nutsedge") is None)
check("real Crabgrass pool holds no bank crops",
      not [e for e in D._class_image_pool("Crabgrass") if e["kind"] == "bank"])
check("MorningGlory reads the Crabgrass folder",
      [e["fname"] for e in D._class_image_pool("MorningGlory") if e["kind"] == "bank"]
      == ["cottonweeddet12_Crabgrass_0.png"])
check("Ragweed reads the Nutsedge folder, real Nutsedge does not",
      [e["dir"] for e in D._class_image_pool("Ragweed") if e["kind"] == "bank"] == ["Nutsedge"]
      and not [e for e in D._class_image_pool("Nutsedge") if e["kind"] == "bank"])
check("landing summary for real Crabgrass counts no bank crops",
      D._class_summary_landing("Crabgrass")["n_bank"] == 0
      and D._class_summary_landing("MorningGlory")["n_bank"] == 1)

# v3.60.0: once the species bank is built the pages show it (the bank the
# pipeline reads), keyed 'banksp/' so no key is read against the other bank
sbank = TMP / "results" / "framework" / "synth_cutpaste" / "object_bank_species"
(sbank / "Ragweed").mkdir(parents=True, exist_ok=True)
(sbank / ".vocabulary").write_text("species\n")
(sbank / "Ragweed" / "cottonweed_sp8_r_0000.png").write_bytes(b"\x89PNG")
check("the species bank is shown once it is marked", D._bank_root() == D._SPECIES_BANK_DIR)
pool = [e for e in D._class_image_pool("Ragweed") if e["kind"] in ("bank", "banksp")]
check("species-bank crops are pooled as 'banksp' from the species folder",
      [(e["kind"], e["dir"]) for e in pool] == [("banksp", "Ragweed")], pool)
key = D._pool_entry_urls(pool[0], "Ragweed")[0] if pool else ""
check("their exemplar key names the species bank", key == "banksp/Ragweed/cottonweed_sp8_r_0000.png", key)
check("the thumbnail source resolves in the species bank",
      D._source_for("banksp", "Ragweed", "cottonweed_sp8_r_0000.png") is not None
      and D._source_for("bank", "Ragweed", "cottonweed_sp8_r_0000.png") is None)
check("a species-era FLUX file is served from synth_diffusion_species",
      D._flux_img_path("fluxsp_Ragweed_000000.jpg").parent == D._FLUX_SP_IMG_DIR
      and D._flux_img_path("fluxsynth_Ragweed_000000.jpg").parent == D._FLUX_IMG_DIR)
(sbank / ".vocabulary").unlink()

# v3.60.0: a cwd12 copy's AI review cached before v3.60.0 is regenerated
D._DATASET_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
stale = {"ok": True, "summary": "Carpetweeds are 16x more frequent than SpottedSpurge"}
(D._DATASET_ANALYSIS_DIR / "cottonweed_sp8.ai.json").write_text(json.dumps(stale))
(D._DATASET_ANALYSIS_DIR / "rf_zig.ai.json").write_text(json.dumps(stale))
check("a pre-v3.60.0 AI review of a cwd12 copy is not served from cache",
      D._ai_review_prepare("cottonweed_sp8").get("cached") != stale)
check("another slug's cached review is served as is",
      D._ai_review_prepare("rf_zig").get("cached") == stale)

print("OWL upload button")
argv = D._CLUSTER_ACTIONS["owl_upload_proposals"]["argv"]
check("the upload button names the species owl_preannotate_one produces",
      argv[argv.index("--species") + 1] == "SpottedSpurge", argv)

print("static 12-class table")
from weed_optimizer_framework.tools import dashboard_generator as G  # noqa: E402
check("generator slots named by the species they hold",
      G.CANONICAL_12 == S.TRAINER_SLOT_SPECIES)
html = G.build_categories({"crop_counts": {}, "source_counts": {}, "annotation_counts": {},
                           "twelve_class_gt": {"Waterhemp": 7}})
check("12-class table shows species with legacy label second",
      "<td>0</td><td>Waterhemp</td><td>Carpetweeds</td><td>7</td>" in html)

check("the dashboard does not import mega_trainer",
      "weed_optimizer_framework.tools.mega_trainer" not in sys.modules)

import shutil  # noqa: E402
shutil.rmtree(TMP, ignore_errors=True)

if FAILURES:
    print("\n%d FAILED: %s" % (len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("\nall passed")
