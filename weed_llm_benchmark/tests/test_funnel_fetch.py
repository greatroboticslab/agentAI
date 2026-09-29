#!/usr/bin/env python3
"""Lab fetches (docs/FUNNEL_AUDIT.md §6 H12, §7 R-F, §8.5 L11a/L12, DEC-7, DEC-8;
runner §4.5, §5.2.5).

What is pinned, and why:
  * cards: every card resolver's specs are fetched through the transport and
    written with their sha256; a Mendeley file must match the sha256 the
    listing publishes and a Zenodo file its md5 (a mismatch refuses); an
    archive's members are listed with their sha256; an HTTP error of a card is
    recorded, not raised; a Roboflow class list without an API key is refused
    and recorded; the key never reaches a file;
  * KT7 (DEC-8): only CC-licensed photos of research-grade observations, at
    most per_taxon_max per taxon, each with observation id, licence and photo
    sha256; roles by the rule of runner §4.6 (recomputed here from the seed);
    the crop table covers every photo;
  * the manifest: every fetched file is listed with its sha256; the check on
    arrival passes, then refuses a changed or a missing file; the
    machine-local crop table is left out and rebuilt;
  * H12 known items: from literature notes, the surveys they cite (fetched and
    hashed) and the index the config names; a candidate needs a target name
    and boxes; slug patterns come from the config's templates; the list and
    its sources are hashed;
  * taxonomy: the names of a pool summary's joins, resolved through the
    transport (the recorded GBIF answers);
  * R-F refetch: only the upstream images the geometry match left unpaired
    are written, with their upstream labels, a manifest and a crop table;
  * no real network is touched: the socket is blocked.

Run:  python3 tests/test_funnel_fetch.py
"""
import copy
import csv
import hashlib
import io
import json
import os
import pathlib
import re
import shutil
import socket
import sys
import tempfile
import zipfile
import funnel_prereg as FPR  # noqa: E402

TMP = pathlib.Path(tempfile.mkdtemp(prefix="funnel_fetch_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
os.environ.pop("ROBOFLOW_API_KEY", None)
TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))
(TMP / "repo" / "docs").mkdir(parents=True)
FD = TMP / "inc" / "funnel"
FD.mkdir(parents=True)
shutil.copyfile(ROOT.parent / "docs" / "FUNNEL_AUDIT.md", TMP / "repo" / "docs" / "FUNNEL_AUDIT.md")
FPR.write_pre_draw(FD / "prereg_v1.json", ROOT / "results" / "framework" / "inc" / "funnel" / "prereg_v1.json")


class _NoNet(socket.socket):
    def __init__(self, *a, **k):
        raise AssertionError("the test touched the network")


socket.socket = _NoNet

import numpy as np  # noqa: E402

import funnel_world as FW  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.funnel import FetchError, StaleInput, read_json  # noqa: E402
from weed_optimizer_framework.tools.funnel import domain as D  # noqa: E402
from weed_optimizer_framework.tools.funnel import fetch as F  # noqa: E402
from weed_optimizer_framework.tools.funnel import taxonomy as T  # noqa: E402

FAILURES, SKIPS = [], []
try:
    from PIL import Image
except ImportError:
    Image = None


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:1500]))
        FAILURES.append(name)


def raises(fn, exc):
    try:
        fn()
    except exc as e:
        return str(e) or type(e).__name__
    return None


def png(color=(10, 200, 30), size=(12, 9)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def zbytes(members):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, d in members.items():
            z.writestr(n, d)
    return buf.getvalue()


def sha(b):
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------------ the fake web
class Web(object):
    """A fake transport: {(url, frozen params): (status, bytes)}; records calls."""

    def __init__(self):
        self.routes = {}
        self.calls = []

    def add(self, url, body, status=200, params=None, headers=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.routes[(url, json.dumps(params or {}, sort_keys=True))] = (status, data, headers or {})

    def __call__(self, url, params):
        self.calls.append((url, dict(params or {})))
        got = self.routes.get((url, json.dumps(dict(params or {}), sort_keys=True)))
        if got is None:
            return 404, b"not found", {}
        return got[0], got[1], dict(got[2])


ANN_ZIP = zbytes({"ann/PASCAL_VOC/f1.xml": "<annotation><filename>V.mp4_1.png</filename><size><width>10</width>"
                                           "<height>10</height></size></annotation>",
                  "ann/YOLO_darknet/f1.txt": "0 0.5 0.5 0.2 0.2\n"})
LABEL_A = b"0 0.47 0.2 0.06 0.08\n"
CLASSES = b"alpha\nbeta\n"


def build_web(web, dom):
    # Hugging Face card
    web.add(F.HF_API % "Org/card_ok", {"cardData": {"license": "cc-by-4.0"}, "sha": "abc"})
    web.add(F.HF_FILE % ("Org/card_ok", "README.md"), b"# card\nclasses: alpha, beta\n")
    web.add(F.HF_API % "Org/card_denied", {"error": "unauthorized"}, status=401)
    web.add(F.HF_FILE % ("Org/card_denied", "README.md"), b"", status=401)
    # Mendeley dataset with one annotation zip in a folder and label files in two folders
    web.add(F.MENDELEY_DATASET % "mdl1", {"id": "mdl1", "data_licence": {"short_name": "CC BY 4.0"}},
            params={"version": "2"})
    web.add(F.MENDELEY_FOLDERS % ("mdl1", 2), [{"id": "top", "name": "Top"},
                                                {"id": "fa", "name": "Folder A", "parent_id": "top"},
                                                {"id": "fb", "name": "Folder B", "parent_id": "top"}])

    def fentry(name, data, fid, bad=False):
        url = "https://data.mendeley.com/public-files/datasets/mdl1/files/%s/file_downloaded" % name.replace(" ", "_")
        web.add(url, data)
        return {"filename": name, "size": len(data), "content_details": {
            "download_url": url, "sha256_hash": ("0" * 64 if bad else sha(data))}, "folder_id": fid}
    web.add(F.MENDELEY_FILES % "mdl1", [], params={"folder_id": "root", "version": "2"})
    web.add(F.MENDELEY_FILES % "mdl1", [fentry("ann.zip", ANN_ZIP, "top"), fentry("bad.zip", b"xx", "top", bad=True)],
            params={"folder_id": "top", "version": "2"})
    web.add(F.MENDELEY_FILES % "mdl1", [fentry("classes.txt", CLASSES, "fa"), fentry("a1.txt", LABEL_A, "fa"),
                                        fentry("a1.JPG", b"jpegbytes", "fa")],
            params={"folder_id": "fa", "version": "2"})
    web.add(F.MENDELEY_FILES % "mdl1", [fentry("classes.txt", CLASSES, "fb"), fentry("b1.txt", LABEL_A, "fb")],
            params={"folder_id": "fb", "version": "2"})
    # Zenodo record with an md5-checked file
    zfile = b"readme text"
    web.add(F.ZENODO_RECORD % "777", {"metadata": {"license": {"id": "cc-by-4.0"}}, "files": [
        {"key": "README.txt", "checksum": "md5:%s" % hashlib.md5(zfile).hexdigest(),
         "links": {"self": "https://zenodo.org/api/records/777/files/README.txt/content"}}]})
    web.add("https://zenodo.org/api/records/777/files/README.txt/content", zfile)
    # Roboflow project
    web.add(F.ROBOFLOW_PROJECT % ("ws", "proj"), {"project": {"classes": {"weed": 10}}},
            params={"api_key": "SECRET-KEY-123"})
    # a paper page
    web.add("https://example.org/paper", b"<html>paper</html>")


def test_config(dom):
    raw = copy.deepcopy(dom.raw)
    raw["sources"]["card_resolvers"] = {
        "slug_hf": {"fetch": [{"kind": "http", "provider": "huggingface", "repo": "Org/card_ok", "what": "card"},
                              {"kind": "http", "provider": "url", "url": "https://example.org/paper", "what": "paper"}],
                    "class_table": None, "table_source": "t", "upstream_annotations": None},
        "slug_denied": {"fetch": [{"kind": "http", "provider": "huggingface", "repo": "Org/card_denied"}],
                        "class_table": None, "table_source": "t", "upstream_annotations": None},
        "slug_mdl": {"fetch": [{"kind": "http", "provider": "mendeley", "dataset": "mdl1", "version": 2,
                                "what": "card"}],
                     "class_table": None, "table_source": "t",
                     "upstream_annotations": {"kind": "archive", "provider": "mendeley", "dataset": "mdl1", "version": 2,
                                              "what": "annotations", "files": ["ann.zip"], "format": "voc+yolo"}},
        "slug_txt": {"fetch": [], "class_table": None, "table_source": "t",
                     "upstream_annotations": {"kind": "archive", "provider": "mendeley", "dataset": "mdl1",
                                              "version": 2, "what": "annotations", "files_regex": "\\.txt$",
                                              "folder_regex": "Folder", "format": "yolo",
                                              "classes_file": "classes.txt"}},
        "slug_zen": {"fetch": [{"kind": "archive", "provider": "zenodo", "record": "777", "what": "card",
                                "files": ["README.txt"]}], "class_table": None, "table_source": "t",
                     "upstream_annotations": None},
        "slug_rf": {"fetch": [{"kind": "roboflow_classes", "workspace": "ws", "project": "proj",
                               "what": "class_list", "api_key_env": "ROBOFLOW_API_KEY"}],
                    "class_table": None, "table_source": "t", "upstream_annotations": None},
    }
    kt7 = raw["known_truth"]["kt7"]
    kt7["taxa"] = ["Amaranthus palmeri", "Ipomoea", "Amaranthus retroflexus"]
    kt7["per_taxon_max"] = 3
    kt7["provider"]["params"]["per_page"] = "5"
    kt7["roles"] = {"exemplar": 1, "g0_every": 2}
    raw["sources"]["known_items"]["indexes"] = [{"kind": "http", "provider": "url", "what": "index",
                                                 "format": "task_class_index", "url": "https://example.org/index.json",
                                                 "slug_template": "project_agml__{name}"}]
    p = TMP / "weed_test.json"
    p.write_text(json.dumps(raw))
    return D.load(str(p))


# ------------------------------------------------------------------ tests
def test_cards(dom, web):
    print("cards")
    msg = raises(lambda: F.fetch_cards(dom, FD, web), FetchError)
    check("fetch_cards runs through every resolver", msg is None, msg)
    idx = read_json(FD / "cards" / "index.json")
    cards = idx["cards"]
    check("cards/index.json has the funnel header", idx["format"] == "funnel-cards/1" and idx["prereg"]["core_sha256"])
    ok = [e for es in cards.values() for e in es if e.get("file")]
    check("every written card file hashes as recorded", ok and all(
        C.sha256_file(FD / "cards" / e["file"]) == e["sha256"] for e in ok))
    hf = cards["slug_hf"]
    check("the Hugging Face card and README are fetched, licence from the card data",
          [e["what"] for e in hf] == ["card", "card", "paper"] and hf[0]["licence"] == "cc-by-4.0", hf)
    den = cards["slug_denied"]
    check("an HTTP error of a card is recorded, not raised", den and den[0]["status"] == 401 and den[0]["file"] is None,
          den)
    mdl = cards["slug_mdl"]
    zipent = [e for e in mdl if e["what"] == "annotations"]
    check("the Mendeley annotation archive is saved under its name, licence from the dataset",
          len(zipent) == 1 and zipent[0]["file"] == "slug_mdl/ann.zip" and zipent[0]["licence"] == "CC BY 4.0", zipent)
    check("its members are listed with their sha256", zipent[0]["members"] and all(
        len(m["sha256"]) == 64 for m in zipent[0]["members"]) and len(zipent[0]["members"]) == 2)
    txt = cards["slug_txt"]
    files = sorted(e["file"] for e in txt)
    check("a regex spec takes every matching label file under its folder path, never the images",
          files == ["slug_txt/annotations/Top/Folder_A/a1.txt", "slug_txt/annotations/Top/Folder_A/classes.txt",
                    "slug_txt/annotations/Top/Folder_B/b1.txt", "slug_txt/annotations/Top/Folder_B/classes.txt"], files)
    check("the Zenodo file passes its md5 and is saved", any(e["file"] == "slug_zen/README.txt" for e in cards["slug_zen"]))
    check("a Roboflow class list without an API key is refused and recorded",
          cards["slug_rf"][0]["status"] == "refused" and idx["refused"][0]["source"] == "slug_rf")
    os.environ["ROBOFLOW_API_KEY"] = "SECRET-KEY-123"
    try:
        idx2 = F.fetch_cards(dom, FD, web)
    finally:
        os.environ.pop("ROBOFLOW_API_KEY", None)
    rf = idx2["cards"]["slug_rf"][0]
    check("with the key, the class list is fetched", rf["status"] == 200 and rf["file"], rf)
    text = (FD / "cards" / "index.json").read_text()
    check("the API key never reaches a file", "SECRET-KEY-123" not in text and "redacted" in rf["url"], rf["url"])
    bad = copy.deepcopy(dom.raw)
    bad["sources"]["card_resolvers"]["slug_mdl"]["upstream_annotations"]["files"] = ["bad.zip"]
    p = TMP / "weed_bad.json"
    p.write_text(json.dumps(bad))
    msg = raises(lambda: F.fetch_cards(D.load(str(p)), TMP / "badcards", web, prereg=FD / "prereg_v1.json"),
                 FetchError)
    check("a Mendeley file whose bytes differ from the published sha256 refuses", msg is not None and "sha256" in msg, msg)
    unk = copy.deepcopy(dom.raw)
    unk["sources"]["card_resolvers"] = {"x": {"fetch": [{"kind": "ftp", "url": "ftp://x"}]}}
    p2 = TMP / "weed_unk.json"
    p2.write_text(json.dumps(unk))
    check("an unknown fetch kind refuses",
          raises(lambda: F.fetch_cards(D.load(str(p2)), TMP / "unk", web, prereg=FD / "prereg_v1.json"), FetchError))


def inat_obs(oid, pid, licence="cc-by", grade="research"):
    return {"id": oid, "quality_grade": grade, "observed_on": "2024-06-01", "place_guess": "Somewhere",
            "photos": [{"id": pid, "license_code": licence,
                        "url": "https://static.example.org/photos/%d/square.jpg" % pid}]}


def test_kt7(dom, web):
    print("KT7")
    if Image is None:
        print("SKIP: PIL missing")
        SKIPS.append("kt7 (PIL)")
        return
    prov = dom.raw["known_truth"]["kt7"]["provider"]
    per_taxon = {"Amaranthus palmeri": [inat_obs(10, 110), inat_obs(11, 111, licence=None), inat_obs(12, 112),
                                        inat_obs(13, 113, grade="needs_id"), inat_obs(14, 114)],
                 "Ipomoea": [inat_obs(20 + i, 120 + i) for i in range(5)],
                 "Amaranthus retroflexus": [inat_obs(30, 130, licence="cc-by-nc"), inat_obs(31, 131)]}
    for taxon, obs in per_taxon.items():
        for o in obs:
            o["taxon"] = {"name": taxon if taxon != "Ipomoea" else "Ipomoea purpurea"}
    # a name query can return other taxa (iNaturalist's answer for a homonym genus holds other grasses)
    off = inat_obs(19, 119)
    off["taxon"] = {"name": "Cynodon dactylon"}
    notax = inat_obs(18, 118)
    per_taxon["Ipomoea"][:0] = [off, notax]
    for taxon, obs in per_taxon.items():
        params = dict(prov["params"], taxon_name=taxon, page="1")
        web.add(prov["url"], {"total_results": len(obs), "results": obs}, params=params)
        params2 = dict(params, page="2")
        web.add(prov["url"], {"total_results": len(obs), "results": []}, params=params2)
        for o in obs:
            p = o["photos"][0]
            web.add(p["url"].replace("/square.", "/large."), png((p["id"] % 250, 90, 40)))
    out = F.fetch_kt7(dom, FD, web)
    items = [json.loads(ln) for ln in (FD / "kt7" / "kt7_items.jsonl").read_text().splitlines()]
    check("at most per_taxon_max (3) per taxon", out["per_taxon"] == {"Amaranthus palmeri": 3, "Ipomoea": 3,
                                                                      "Amaranthus retroflexus": 2}, out["per_taxon"])
    check("only CC-licensed photos of research-grade observations",
          all(i["licence"] in dom.raw["known_truth"]["kt7"]["licences"] and i["quality_grade"] == "research" for i in items)
          and "t7:11/111" not in {i["id"] for i in items} and "t7:13/113" not in {i["id"] for i in items})
    check("every photo is saved with the sha256 its row records", all(
        C.sha256_file(FD / "kt7" / "photos" / i["file"]) == i["sha256"] for i in items))
    check("each row records the observation id, licence and query", all(
        i["observation_id"] and i["licence"] and i["query"]["taxon_name"] == i["taxon"] for i in items))
    check("the photo is the configured size, not the square thumbnail", all("/large." in i["url"] for i in items))
    check("an observation whose own taxon is not the queried one (or one below it), or has none, is left out; "
          "the observed taxon is recorded", "t7:19/119" not in {i["id"] for i in items}
          and "t7:18/118" not in {i["id"] for i in items} and out["skipped"]["taxon"] == 2
          and all(i["observed_taxon"] == i["taxon"] or i["observed_taxon"].startswith(i["taxon"] + " ")
                  for i in items), (out["skipped"], [i["id"] for i in items]))
    check("targets and attractors are named", {i["target"] for i in items if i["taxon"] == "Amaranthus palmeri"}
          == {"PalmerAmaranth"} and {i["attractor"] for i in items if i["taxon"] == "Amaranthus retroflexus"} == {"A01"})
    ok = True
    for taxon in per_taxon:
        rows = sorted([i for i in items if i["taxon"] == taxon], key=lambda r: r["id"])
        perm = np.random.default_rng(C.stable_int("funnel/v1/kt7/roles/%s" % taxon)).permutation(len(rows))
        for pos, ix in enumerate(perm.tolist()):
            want = "exemplar" if pos < 1 else ("g0" if (pos - 1) % 2 == 1 else "sentinel")
            ok = ok and rows[ix]["role"] == want
    check("roles follow the rule (seeded permutation; exemplars first; every g0_every-th of the rest is g0)", ok)
    with open(FD / "kt7" / "crops_kt7.csv", newline="") as fh:
        tab = list(csv.DictReader(fh))
    check("crops_kt7.csv: one whole-photo row per item, crop ids 0..n-1 in id order",
          [r["key"] for r in tab] == sorted(i["id"] for i in items) and [int(r["crop_id"]) for r in tab]
          == list(range(len(items))) and all(r["w"] == "1.000000" and r["set"] == "kt7" for r in tab))
    msg = raises(lambda: F.fetch_kt7(dom, TMP / "kt7_fail", lambda u, p: (500, b"", {}),
                                     prereg=FD / "prereg_v1.json"), FetchError)
    check("a provider error refuses the fetch", msg is not None and "500" in msg, msg)


def test_manifest():
    print("manifest")
    F.write_manifest(FD)
    man = read_json(FD / F.MANIFEST_NAME)
    check("the manifest lists the fetched files with their sha256", man["files"] and all(
        C.sha256_file(FD / rel) == s for rel, s in man["files"].items()))
    check("the machine-local crop table is left out", "kt7/crops_kt7.csv" not in man["files"])
    os.unlink(FD / "kt7" / "crops_kt7.csv")
    F.check_manifest(FD)
    check("the arrival check passes and rebuilds the machine-local table", (FD / "kt7" / "crops_kt7.csv").exists())
    victim = sorted(r for r in man["files"] if r.startswith("cards/"))[0]
    keep = (FD / victim).read_bytes()
    (FD / victim).write_bytes(keep + b"x")
    msg = raises(lambda: F.check_manifest(FD), StaleInput)
    check("a changed file is detected", msg is not None and victim in msg, msg)
    os.unlink(FD / victim)
    check("a missing file is detected", raises(lambda: F.check_manifest(FD), StaleInput) is not None)
    (FD / victim).write_bytes(keep)
    F.check_manifest(FD)
    check("restored, the check passes again", True)


NOTE = """# {title}
<!-- generated by inc_autopilot/corpus.py build; do not hand-edit -->
id: {nid}
bib: {title}. Journal, 2024. {url}
topics: {topic}
status: weeds=OK|yes|background

## Passages
RESULT [weeds]: {text}
"""


def test_known_items(dom, web):
    print("H12 known items")
    lit = TMP / "literature"
    lit.mkdir()
    (lit / "a.md").write_text(NOTE.format(title="An open dataset of field images (FieldSet1)", nid="a",
                                          url="https://www.kaggle.com/datasets/user1/fieldset1", topic="weeds",
                                          text="The dataset has Palmer amaranth and cotton with bounding boxes."))
    (lit / "b.md").write_text(NOTE.format(title="A classification dataset (ClassSet)", nid="b",
                                          url="https://example.org/b", topic="weeds",
                                          text="The dataset holds Waterhemp images labelled by folder."))
    (lit / "c.md").write_text(NOTE.format(title="A method paper", nid="c", url="https://example.org/c",
                                          topic="attribution", text="Palmer amaranth dataset bounding boxes."))
    (lit / "d.md").write_text(NOTE.format(title="An updated survey of public datasets", nid="d",
                                          url="https://github.com/someone/Survey-Repo", topic="weeds",
                                          text="A source list of datasets."))
    readme = ("# Survey\n\nSmith (2023). Detection of waterhemp in soybean. Journal.\n"
              "[[paper]](https://example.org/p1) [[dataset]](https://universe.roboflow.com/ws1/proj-1)\n\n"
              "Doe (2022). Crop rows. [[dataset]](https://example.org/d2)\n\ndataset info: 10 images\n")
    web.add("https://raw.githubusercontent.com/someone/Survey-Repo/main/README.md", readme.encode("utf-8"))
    web.add("https://example.org/index.json", {
        "sick_detection": {"ml_task": "object_detection", "classes": {"0": "sicklepod", "1": "soil"},
                           "docs_url": "https://example.org/sick"},
        "sick_classification": {"ml_task": "image_classification", "classes": {"0": "sicklepod"}},
        "maize_detection": {"ml_task": "object_detection", "classes": {"0": "maize"}}})
    out = FD / "known_items_v1.json"
    doc = F.known_items([lit], out, domain=dom, transport=web)
    names = {i["name"]: i for i in doc["items"]}
    check("known_items_v1.json has the funnel header, the rule and the hashed sources",
          doc["format"] == "funnel-known-items/1" and doc["rule"] and all(
              ("sha256" in s) or ("status" in s) for s in doc["sources_used"]))
    check("a dataset note naming a target with boxes is an item, its slug pattern from the config's template",
          "FieldSet1" in names and "^kg_user1__fieldset1$" in names["FieldSet1"]["slug_patterns"]
          and names["FieldSet1"]["species_named"] == ["PalmerAmaranth"], names.get("FieldSet1"))
    check("a note without box words is not", "ClassSet" not in names)
    check("a note outside the domain's topics is not", "A method paper" not in names)
    sv = [i for i in doc["items"] if i["doc_ref"].startswith("survey:")]
    check("the survey the note cites is fetched and its dataset entries parsed, named by the dataset link",
          len(sv) == 1 and sv[0]["species_named"] == ["Waterhemp"] and sv[0]["name"] == "universe.roboflow.com/ws1/proj-1"
          and [p for p in sv[0]["slug_patterns"] if re.search(p, "rf_ws1__proj-1")] == sv[0]["slug_patterns"], sv)
    check("the index's detection entry naming a target is an item, with the index's slug template",
          "sick_detection" in names and "^project_agml__sick_detection$" in names["sick_detection"]["slug_patterns"])
    check("the index's classification entry and target-free entry are not",
          "sick_classification" not in names and "maize_detection" not in names)
    check("the fetched survey and index are saved and hashed",
          (FD / "known_items_sources").is_dir() and any(s.get("ref", "").endswith("index.json") and s.get("sha256")
                                                          for s in doc["sources_used"]))
    doc2 = F.known_items([lit], FD / "known_items_again.json", domain=dom, transport=web)
    check("the list is deterministic", [i for i in doc2["items"]] == doc["items"])


def test_taxonomy(web_unused):
    print("taxonomy through fetch")
    names_from = TMP / "pool_summary_like.json"
    names_from.write_text(json.dumps({"per_slug": {"a": {"join": {"0": ["Redroot Pigweed", "OtherPlant"],
                                                                  "1": ["Waterhemp", "Waterhemp"]}},
                                                   "b": {"join": {"*": [None, "OtherPlant"]}}}}))
    cache = F.taxonomy("weed", names_from, FD / "taxonomy_cache.json", FW.replay_transport())
    check("the names of the joins are resolved and the cache written",
          "redrootpigweed" in cache["resolved"] and "waterhemp" in cache["resolved"]
          and T.load_cache(FD / "taxonomy_cache.json", D.load("weed"))["resolved"] == cache["resolved"])
    check("the names file is recorded as the cache's input", cache["inputs"]["names_from"]["sha256"]
          == C.sha256_file(names_from))
    bad = TMP / "bad_names.json"
    bad.write_text(json.dumps({"x": 1}))
    check("a file with neither names nor joins refuses",
          raises(lambda: F.taxonomy("weed", bad, FD / "t2.json", FW.replay_transport()), FetchError))


def test_refetch(dom, web):
    print("R-F refetch")
    if Image is None:
        print("SKIP: PIL missing")
        SKIPS.append("refetch (PIL)")
        return
    slug = "slug_mh"
    raw = copy.deepcopy(dom.raw)
    stems = ["s%02d" % i for i in range(4)]
    ann = {}
    imgs = {}
    for i, s in enumerate(stems):
        ann["x/PASCAL_VOC/%s.xml" % s] = ("<annotation><filename>V.mp4_%d.png</filename><size><width>12</width>"
                                          "<height>9</height></size><object><name>n%d</name><bndbox><xmin>1</xmin>"
                                          "<ymin>1</ymin><xmax>6</xmax><ymax>5</ymax></bndbox></object></annotation>"
                                          % (i, i))
        ann["x/YOLO_darknet/%s.txt" % s] = "%d 0.291667 0.333333 0.416667 0.444444\n" % i
        imgs["imgs/%s.png" % s] = png((i * 40, 10, 10))
    cdir = TMP / "rf" / "cards" / slug
    cdir.mkdir(parents=True)
    (cdir / "ann.zip").write_bytes(zbytes(ann))
    (TMP / "rf" / "cards" / "index.json").write_text(json.dumps({"cards": {slug: [
        {"what": "annotations", "file": "%s/ann.zip" % slug, "sha256": C.sha256_file(cdir / "ann.zip")}]}}))
    (TMP / "rf" / "relation_geometry_v1.json").write_text(json.dumps({"matches": {slug: {
        "paired_upstream": stems[:2]}}}))
    shutil.copyfile(FD / "prereg_v1.json", TMP / "rf" / "prereg_v1.json")
    izip = zbytes(imgs)
    url = "https://data.mendeley.com/public-files/datasets/mdl9/files/imgs/file_downloaded"
    web.add(F.MENDELEY_DATASET % "mdl9", {"data_licence": {"short_name": "CC BY 4.0"}}, params={"version": "1"})
    web.add(F.MENDELEY_FOLDERS % ("mdl9", 1), [])
    web.add(F.MENDELEY_FILES % "mdl9", [{"filename": "imgs.zip", "content_details": {"download_url": url,
                                                                                     "sha256_hash": sha(izip)}}],
            params={"folder_id": "root", "version": "1"})
    web.add(url, izip)
    raw["sources"]["card_resolvers"][slug] = {
        "fetch": [], "class_table": None, "table_source": "t",
        "upstream_annotations": {"kind": "archive", "provider": "mendeley", "dataset": "mdl9", "version": 1,
                                 "files": ["ann.zip"], "format": "voc+yolo", "voc_dir": "PASCAL_VOC",
                                 "yolo_dir": "YOLO_darknet"},
        "refetch_images": {"kind": "archive", "provider": "mendeley", "dataset": "mdl9", "version": 1,
                           "files": ["imgs.zip"]}}
    p = TMP / "weed_refetch.json"
    p.write_text(json.dumps(raw))
    out = F.refetch(D.load(str(p)), slug, TMP / "rf", web)
    man = read_json(TMP / "rf" / "refetch" / "manifest.json")
    got = sorted(r["stem"] for r in man["images"])
    check("only the upstream images the geometry match left unpaired are written", got == stems[2:] and out["images"] == 2,
          got)
    check("each image and its upstream label hash as recorded", all(
        C.sha256_file(TMP / "rf" / r["image"]) == r["sha256"] and C.sha256_file(TMP / "rf" / r["label"])
        == r["label_sha256"] for r in man["images"]))
    check("the label keeps the upstream id", (TMP / "rf" / man["images"][0]["label"]).read_text().startswith("2 "))
    check("the manifest says the Step 1 chain has not run and records the licence",
          man["chain"].startswith("not run") and man["licence"] == "CC BY 4.0")
    with open(TMP / "rf" / "refetch" / "crops_refetch.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    check("crops_refetch.csv has one row per upstream box", len(rows) == 2 and {r["key"] for r in rows} == set(stems[2:]))
    os.unlink(TMP / "rf" / "relation_geometry_v1.json")
    check("refetch refuses without the geometry match",
          raises(lambda: F.refetch(D.load(str(p)), slug, TMP / "rf", web), FetchError))


def test_partial_listing(web):
    """The Mendeley public API lists at most 1,000 files per folder and answers
    206 with "Content-Range: items a-b/total" (it pages by no parameter). A
    partially listed folder outside the spec's folder_regex is recorded; one
    inside the spec's scope refuses the fetch."""
    print("partial Mendeley listings")
    web.add(F.MENDELEY_FOLDERS % ("mdl2", 1), [{"id": "lab", "name": "Label files"}, {"id": "crp", "name": "Crops"}])
    lab = b"0 0.5 0.5 0.1 0.1\n"
    url = "https://data.mendeley.com/public-files/datasets/mdl2/files/l1/file_downloaded"
    web.add(url, lab)
    web.add(F.MENDELEY_FILES % "mdl2", [], params={"folder_id": "root", "version": "1"})
    web.add(F.MENDELEY_FILES % "mdl2", [{"filename": "l1.txt", "content_details": {"download_url": url,
                                                                                   "sha256_hash": sha(lab)}}],
            params={"folder_id": "lab", "version": "1"})
    web.add(F.MENDELEY_FILES % "mdl2", [{"filename": "c%d.jpg" % i, "content_details": {}} for i in range(2)],
            status=206, params={"folder_id": "crp", "version": "1"}, headers={"Content-Range": "items 0-1/5"})
    out = TMP / "partial" / "cards"
    spec = {"kind": "archive", "provider": "mendeley", "dataset": "mdl2", "version": 1, "what": "annotations",
            "files_regex": "\\.txt$", "folder_regex": "^Label files$", "format": "yolo"}
    ents = F.fetch_spec(spec, "slug_p", out, web)
    listing = read_json(out / "slug_p" / "mendeley_mdl2_v1_files.json")
    check("a 206 folder outside folder_regex is recorded as partial, and the label file is fetched",
          [e["file"] for e in ents] == ["slug_p/annotations/Label_files/l1.txt"]
          and listing["partial_folders"] == {"Crops": {"listed": 2, "total": 5, "content_range": "items 0-1/5"}},
          (ents, listing.get("partial_folders")))
    wide = dict(spec)
    del wide["folder_regex"]
    msg = raises(lambda: F.fetch_spec(wide, "slug_p", TMP / "partial2" / "cards", web), FetchError)
    check("a 206 folder inside the spec's scope refuses the fetch, naming the folder",
          msg is not None and "Crops" in msg and "folder_regex" in msg, msg)


def test_roboflow_key_file(web):
    """Without the key in the environment, the Roboflow class list reads the
    key file the spec names (the platform's own key file), records the
    project's licence, and never saves the key."""
    print("Roboflow key file and licence")
    web.add(F.ROBOFLOW_PROJECT % ("ws", "lic"), {"project": {"classes": {"0": 5}, "license": "CC BY 4.0"}},
            params={"api_key": "FILE-KEY-456"})
    web.add(F.ROBOFLOW_PROJECT % ("ws", "leaky"), {"project": {"note": "FILE-KEY-456"}},
            params={"api_key": "FILE-KEY-456"})
    kf = TMP / "rf_key"
    kf.write_text("FILE-KEY-456\n")
    old = os.environ.pop("ROBOFLOW_API_KEY", None)
    try:
        spec = {"kind": "roboflow_classes", "workspace": "ws", "project": "lic", "what": "class_list",
                "api_key_env": "ROBOFLOW_API_KEY", "api_key_file": str(kf)}
        out = TMP / "rfkey" / "cards"
        ents = F.fetch_spec(spec, "slug_k", out, web)
        text = "".join(pp.read_text() for pp in out.rglob("*") if pp.is_file()) + json.dumps(ents)
        check("the key file is read when the environment has no key; the licence comes from the project",
              ents[0]["status"] == 200 and ents[0]["licence"] == "CC BY 4.0", ents)
        check("  and the key reaches no file and no entry", "FILE-KEY-456" not in text, ents[0]["url"])
        msg = raises(lambda: F.fetch_spec(dict(spec, project="leaky"), "slug_k", TMP / "rfkey2" / "cards", web),
                     FetchError)
        check("an answer that echoes the key is refused and not saved",
              msg is not None and not list((TMP / "rfkey2").rglob("*.json")), msg)
        msg = raises(lambda: F.fetch_spec(dict(spec, api_key_file=str(TMP / "no_such_key")), "slug_k",
                                          TMP / "rfkey3" / "cards", web), FetchError)
        check("no key in the environment or the file: refused, naming both", msg is not None
              and "$ROBOFLOW_API_KEY" in msg and "no_such_key" in msg, msg)
    finally:
        if old is not None:
            os.environ["ROBOFLOW_API_KEY"] = old


if __name__ == "__main__":
    try:
        dom = test_config(D.load("weed"))
        web = Web()
        build_web(web, dom)
        test_cards(dom, web)
        test_kt7(dom, web)
        test_manifest()
        test_known_items(dom, web)
        test_taxonomy(web)
        test_refetch(dom, web)
        test_partial_listing(web)
        test_roboflow_key_file(web)
        check("no call went to the real network (every call went through the fake transport)",
              all(isinstance(u, str) for u, _p in web.calls))
    finally:
        shutil.rmtree(str(TMP), ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "none"))
    sys.exit(1 if FAILURES else 0)
