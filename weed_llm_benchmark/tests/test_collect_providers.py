#!/usr/bin/env python3
"""The collector's providers and transport (docs/CONTINUOUS_LOOP.md §7.2: a byte
cap, a timeout scaled to size and checksum verification where the provider
publishes one), each against a fake network (no socket is opened).

Pinned per provider: what search and describe read (licence and its
evidence, declared classes with box counts, annotation kind, size, image
count), the files listing, and the download's check:
  * Zenodo: the md5 it publishes; a mismatch leaves nothing behind;
  * Mendeley Data: the sha256 of its listing; a folder listed only in part
    (206, Content-Range, not pageable) inside the selection refuses, outside
    it does not; the institutions join the keywords;
  * Hugging Face: ClassLabel names from the card, object-detection tags as
    boxes, LFS sha256 and the git blob sha1 of small files;
  * Kaggle: no token -> CredentialsMissing (card X16); a bearer token or
    kaggle.json basic auth is sent in a header and never recorded;
  * GitHub: release assets with the sha256 digest GitHub publishes, the
    README read for its text;
  * Roboflow: the key is a parameter and never reaches a record; declared
    class box counts; an export still being generated is polled;
  * the annotation index: the CSRF handshake, the aggregation search, the
    per-category box counts and the archive's size from a HEAD request;
  * the record server + FTP: the licence attribute, the class-code
    directories kept by the EPPO table (only a target's code), the published
    sha512 checked, a mismatch refused;
transport: a listed size over the cap refuses before any byte; a stream past
the cap aborts and leaves nothing; the deadline scales with size; URLs are
redacted; secrets come from the environment or a file.

Run:  python3 tests/test_collect_providers.py
"""
import base64
import hashlib
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_collect_world as W  # noqa: E402

TMP = W.setup("collect_providers_")
check, raises = W.check, W.raises
DATA = b"archive bytes " * 100


def md5(b):
    return hashlib.md5(b).hexdigest()


def sha(b, algo="sha256"):
    return hashlib.new(algo, b).hexdigest()


def prov(cfg, name, net, **sec):
    from weed_optimizer_framework.tools.collect import providers as P
    p = P.get(cfg, name, net)
    p.sec.update(sec)
    return p


def test_zenodo(cfg):
    from weed_optimizer_framework.tools.collect import ChecksumMismatch
    print("zenodo")
    rec = {"id": 77, "revision": 3, "links": {"self_html": "https://zenodo.org/records/77"},
           "metadata": {"title": "Boxes of plants", "description": "<p>bounding boxes</p>", "keywords": ["k"],
                        "license": {"id": "cc-by-4.0"}, "resource_type": {"type": "dataset"},
                        "creators": [{"name": "Doe, Jane", "affiliation": "Some University"}]},
           "files": [{"key": "data.zip", "size": len(DATA), "checksum": "md5:%s" % md5(DATA),
                      "links": {"self": "https://zenodo.org/api/records/77/files/data.zip/content"}}]}
    net = W.make_net({("GET", "https://zenodo.org/api/records"): lambda p, d, h: (200, {"hits": {"hits": [rec]}})
                      if p.get("q") == '"x"' else (200, {"hits": {"hits": []}}),
                      ("GET", "https://zenodo.org/api/records/77"): (200, rec),
                      ("GET", "https://zenodo.org/api/records/77/files/data.zip/content"): (200, DATA)})
    z = prov(cfg, "zenodo", net)
    hits = z.search('"x"', 5)
    m = z.describe("77")
    check("search and describe read the record", len(hits) == 1 and m["source_id"] == "zenodo_77"
          and m["licence_text"] == "cc-by-4.0" and m["bytes"] == len(DATA) and m["description"] == "bounding boxes"
          and "Some University" in m["keywords"], m)
    fs = z.files(m)
    check("files carry the published md5", fs[0]["checksums"] == {"md5": md5(DATA)} and fs[0]["size"] == len(DATA))
    r = z.fetch_file(fs[0], TMP / "z" / "a")
    check("a download is checked and recorded", r["sha256"] == sha(DATA) and r["bytes"] == len(DATA)
          and (TMP / "z" / "a").read_bytes() == DATA)
    bad = dict(fs[0], checksums={"md5": "0" * 32})
    e = raises(lambda: z.fetch_file(bad, TMP / "z" / "b"), ChecksumMismatch)
    check("an md5 mismatch refuses and leaves nothing", e is not None and not (TMP / "z" / "b").exists()
          and not (TMP / "z" / "b.tmp").exists(), e)


def test_mendeley(cfg):
    from weed_optimizer_framework.tools.collect import ProviderError
    print("mendeley")
    body = {"id": "abc", "name": "Plants", "description": "boxes", "version": 1,
            "data_licence": {"short_name": "CC BY 4.0"}, "categories": [{"label": "Vision"}],
            "institutions": [{"name": "Some State University"}], "size": 1234}
    dl = "https://data.mendeley.com/public-files/datasets/abc/files/f1/file_downloaded"
    files_a = [{"filename": "imgs.zip", "size": len(DATA), "content_details": {"download_url": dl,
                                                                              "sha256_hash": sha(DATA)}}]
    many = [{"filename": "x%d.jpg" % i, "content_details": {"download_url": dl + str(i)}} for i in range(3)]

    def files(p, d, h):
        if p.get("folder_id") == "root":
            return 200, []
        if p.get("folder_id") == "fa":
            return 200, files_a
        return 206, many, {"Content-Range": "items 0-2/1500"}
    net = W.make_net({("GET", "https://data.mendeley.com/public-api/datasets/abc"): (200, body),
                      ("GET", "https://data.mendeley.com/public-api/datasets/abc/folders/1"):
                          (200, [{"id": "fa", "name": "annotated"}, {"id": "fb", "name": "raw"}]),
                      ("GET", "https://data.mendeley.com/public-api/datasets/abc/files"): files,
                      ("GET", dl): (200, DATA)})
    mp = prov(cfg, "mendeley", net)
    m = mp.describe("abc")
    check("describe reads the licence short name, the version and the institutions", m["ref"] == "abc/1"
          and m["source_id"] == "mendeley_abc_v1" and m["licence_text"] == "CC BY 4.0"
          and "Some State University" in m["keywords"], m)
    e = raises(lambda: mp.files(m), ProviderError)
    check("a folder listed only in part (206) inside the selection refuses", e is not None and "206" in str(e), e)
    fs = mp.files(m, {"folder_regex": "^annotated$"})
    check("outside it the listing is used, with the sha256 it publishes",
          [f["name"] for f in fs] == ["annotated/imgs.zip"] and fs[0]["checksums"] == {"sha256": sha(DATA)}, fs)
    r = mp.fetch_file(fs[0], TMP / "m" / "a")
    check("the download is checked", r["sha256"] == sha(DATA))


def test_huggingface(cfg):
    print("huggingface")
    small = b"readme text"
    blob = hashlib.sha1(b"blob %d\0" % len(small) + small).hexdigest()
    d = {"id": "Owner-X/Plant_Boxes", "sha": "abc123", "tags": ["task_categories:object-detection", "license:mit"],
         "description": "boxes", "cardData": {"license": "cc-by-4.0", "dataset_info": {"features": [
             {"name": "objects", "struct": [{"name": "categories", "list": {"class_label": {"names": {
                 "0": "Palmer amaranth", "1": "soybean"}}}}]}], "splits": [{"name": "train", "num_examples": 40}]}},
         "siblings": [{"rfilename": "README.md", "size": len(small), "blobId": blob},
                      {"rfilename": "data/train-0.parquet", "size": len(DATA), "blobId": "x",
                       "lfs": {"sha256": sha(DATA)}}]}
    base = "https://huggingface.co/datasets/Owner-X/Plant_Boxes/resolve/abc123/"
    net = W.make_net({("GET", "https://huggingface.co/api/datasets/Owner-X/Plant_Boxes"): (200, d),
                      ("GET", "https://huggingface.co/api/datasets"): (200, [d]),
                      ("GET", base + "README.md"): (200, small), ("GET", base + "data/train-0.parquet"): (200, DATA)})
    h = prov(cfg, "huggingface", net)
    m = h.describe("Owner-X/Plant_Boxes")
    check("the source id is the registry's owner__name convention", m["source_id"] == "owner_x__plant_boxes")
    check("ClassLabel names are the declared classes; the detection tag makes boxes",
          [c["name"] for c in m["classes"]] == ["Palmer amaranth", "soybean"] and m["annotation"] == "boxes"
          and m["images"] == 40 and m["licence_text"] == "cc-by-4.0", m)
    fs = {f["name"]: f for f in h.files(m)}
    check("LFS files carry their sha256, others their git blob sha1",
          fs["data/train-0.parquet"]["checksums"] == {"sha256": sha(DATA)}
          and fs["README.md"]["checksums"] == {"git-blob-sha1": blob})
    r1 = h.fetch_file(fs["README.md"], TMP / "h" / "r")
    r2 = h.fetch_file(fs["data/train-0.parquet"], TMP / "h" / "p")
    check("both checks pass on the right bytes", r1["git-blob-sha1"] == blob and r2["sha256"] == sha(DATA))
    check("search reads the same records", h.search("palmer", 5)[0]["source_id"] == "owner_x__plant_boxes")


def test_kaggle(cfg):
    from weed_optimizer_framework.tools.collect import CredentialsMissing
    print("kaggle")
    view = {"ref": "own/set", "title": "Set", "subtitle": "boxes", "description": "yolo labels",
            "licenseName": "CC0: Public Domain", "totalBytes": 99, "currentVersionNumber": 2}
    net = W.make_net({("GET", "https://www.kaggle.com/api/v1/datasets/view/own/set"): (200, view),
                      ("GET", "https://www.kaggle.com/api/v1/datasets/download/own/set"): (200, DATA)})
    k = prov(cfg, "kaggle", net)
    ok, why = k.credentials()
    check("no token: credentials missing, naming where they are looked for (card X16)", not ok and "X16" in why
          and "KAGGLE_API_TOKEN" in why, why)
    check("describe without a token raises CredentialsMissing", raises(lambda: k.describe("own/set"),
                                                                      CredentialsMissing) is not None)
    os.environ["KAGGLE_API_TOKEN"] = "tok-123"
    try:
        m = k.describe("own/set")
        hdr = [x[3] for x in net.log if "view" in x[1]][-1]
        check("a bearer token is sent in a header", hdr.get("Authorization") == "Bearer tok-123")
        check("describe reads the licence name and size; the id is kg_<owner>__<name>",
              m["source_id"] == "kg_own__set" and m["licence_text"] == "CC0: Public Domain" and m["bytes"] == 99)
        fs = k.files(m)
        r = k.fetch_file(fs[0], TMP / "k" / "a", max_bytes=10 ** 6)
        check("the whole-dataset archive is downloaded under the cap", r["bytes"] == len(DATA))
        check("the token reaches no record", "tok-123" not in json.dumps([m, fs, r, net.calls]))
    finally:
        os.environ.pop("KAGGLE_API_TOKEN", None)
    kj = W.TMP / "home" / ".kaggle" / "kaggle.json"
    kj.parent.mkdir(parents=True, exist_ok=True)
    kj.write_text(json.dumps({"username": "u", "key": "k"}))
    try:
        k.describe("own/set")
        hdr = [x[3] for x in net.log if "view" in x[1]][-1]
        check("kaggle.json gives basic auth", hdr.get("Authorization") == "Basic " + base64.b64encode(b"u:k").decode())
    finally:
        kj.unlink()


def test_github(cfg):
    print("github")
    repo = {"full_name": "own/rep", "default_branch": "main", "size": 10, "html_url": "https://github.com/own/rep",
            "description": "plant boxes", "topics": ["dataset"], "license": {"spdx_id": "MIT"}}
    rel = [{"tag_name": "v1", "assets": [{"name": "data.zip", "size": len(DATA),
                                          "browser_download_url": "https://github.com/own/rep/releases/download/v1/data.zip",
                                          "digest": "sha256:%s" % sha(DATA)}]}]
    net = W.make_net({("GET", "https://api.github.com/repos/own/rep"): (200, repo),
                      ("GET", "https://api.github.com/repos/own/rep/readme"):
                          (200, {"content": base64.b64encode(b"YOLO labels of plants").decode()}),
                      ("GET", "https://api.github.com/repos/own/rep/releases"): (200, rel),
                      ("GET", "https://github.com/own/rep/releases/download/v1/data.zip"): (200, DATA)})
    g = prov(cfg, "github", net)
    m = g.describe("own/rep")
    check("describe reads the repository licence (repository level) and the README text",
          m["licence_text"] == "MIT" and "YOLO labels" in m["description"] and m["source_id"] == "gh_own__rep", m)
    fs = g.files(m)
    check("release assets carry GitHub's sha256 digest", fs[0]["checksums"] == {"sha256": sha(DATA)})
    check("the download is checked", g.fetch_file(fs[0], TMP / "g" / "a")["sha256"] == sha(DATA))
    raw = b"raw label file"
    blob = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
    net.route("https://api.github.com/repos/own/rep/contents/labels/a.txt",
              lambda p, d, h: (200, {"size": len(raw), "sha": blob,
                                     "download_url": "https://raw.githubusercontent.com/own/rep/main/labels/a.txt"})
              if p.get("ref") == "main" else (404, {}))
    net.route("https://raw.githubusercontent.com/own/rep/main/labels/a.txt", (200, raw))
    fs = g.files(m, {"raw_paths": ["labels/a.txt"]})
    check("raw files a known item names come with their git blob sha1, and are checked",
          fs[0]["checksums"] == {"git-blob-sha1": blob} and g.fetch_file(fs[0], TMP / "g" / "raw")["bytes"] == len(raw))
    check("the provider is lab only in the config (placement.lab_only)", cfg.lab_only("github") is True
          and not cfg.lab_only("zenodo"))


def test_roboflow(cfg):
    from weed_optimizer_framework.tools.collect import CredentialsMissing
    print("roboflow")
    polls = {"n": 0}

    def export(p, d, h):
        polls["n"] += 1
        return (200, {"progress": 0.5}) if polls["n"] == 1 else (200, {"export": {"link": "https://storage/x.zip"}})
    net = W.make_net({("GET", "https://api.roboflow.com/universe/search"): (200, {"page_size": 12, "results": [
        {"name": "p", "url": "https://universe.roboflow.com/ws-a/proj-b", "workspace": {"url": "ws-a"},
         "type": "object-detection", "classes": ["Palmer Amaranth"], "images": 9, "latestVersion": 3}]}),
        ("GET", "https://api.roboflow.com/ws-a/proj-b"): (200, {"project": {"name": "p", "type": "object-detection",
                                                                         "classes": {"Palmer Amaranth": 120, "soy": 4},
                                                                         "license": "CC BY 4.0", "images": 9},
                                                             "versions": [{"id": "ws-a/proj-b/3"}]}),
        ("GET", "https://api.roboflow.com/ws-a/proj-b/3/yolov8"): export,
        ("GET", "https://storage/x.zip"): (200, DATA)})
    r = prov(cfg, "roboflow", net, export_poll_s=0)
    check("without a key: CredentialsMissing", raises(lambda: r.search("x", 5), CredentialsMissing) is not None)
    os.environ["ROBOFLOW_API_KEY"] = "rf-secret-9"
    try:
        hits = r.search("Palmer amaranth", 5)
        m = r.describe("ws-a/proj-b")
        check("search reads declared classes; the id is rf_<ws>__<proj>", hits[0]["source_id"] == "rf_ws-a__proj-b"
              and hits[0]["classes"][0]["name"] == "Palmer Amaranth")
        check("describe reads class box counts, the licence and the latest version",
              {c["name"]: c["boxes"] for c in m["classes"]} == {"Palmer Amaranth": 120, "soy": 4}
              and m["version"] == "3" and m["licence_text"] == "CC BY 4.0", m)
        fs = r.files(m)
        check("an export being generated is polled until its link comes", polls["n"] == 2 and fs[0]["url"]
              == "https://storage/x.zip")
        r.fetch_file(fs[0], TMP / "r" / "a")
        check("the key never reaches a record (URLs are redacted)", "rf-secret-9" not in json.dumps([hits, m, fs, net.calls])
              and any("<redacted>" in u for u, _s in net.calls), net.calls[:2])
    finally:
        os.environ.pop("ROBOFLOW_API_KEY", None)


def test_weedai(cfg):
    print("annotation index")
    base = cfg.provider("weedai")["base_url"]
    uid = "11111111-2222-3333-4444-555555555555"
    seen = {}

    def msearch(p, d, h):
        seen["body"], seen["headers"] = d, h
        return 200, {"responses": [{"aggregations": {"u": {"buckets": [{"key": uid, "doc_count": 7}]}}}]}
    info = {"metadata": {"name": "Plants", "license": "https://creativecommons.org/licenses/by/4.0/",
                         "creator": [{"name": "a", "affiliation": {"name": "Uni"}}]},
            "agcontexts": [{"n_images": 7, "category_statistics": {"weed: amaranthus palmeri": {
                "image_count": 7, "bounding_box_count": 30}}}], "head_version": 1}
    net = W.make_net({("GET", base + "/api/set_csrf/"): (200, b""),
                      ("POST", base + "/elasticsearch/weedid/_msearch"): msearch,
                      ("GET", base + "/api/upload_info/" + uid): (200, info),
                      ("GET", base + "/code/download/%s.zip" % uid): (200, DATA)})
    net.cookies["csrftoken"] = "tok"
    a = prov(cfg, "weedai", net)
    hits = a.search("Amaranthus palmeri", 8)
    body = seen["body"].decode().splitlines()
    check("the search sends the CSRF token and a match_phrase + aggregation query",
          seen["headers"].get("X-CSRFToken") == "tok" and json.loads(body[1])["query"]["match_phrase"]
          == {"annotations.category.name": "amaranthus palmeri"} and hits[0]["ref"] == uid, body)
    m = a.describe(uid)
    check("describe reads per-category box counts, the taxon hint, the licence URL and the archive size",
          m["classes"][0]["boxes"] == 30 and m["classes"][0]["hints"] == ["amaranthus palmeri"]
          and m["licence_text"].startswith("https://creativecommons") and m["bytes"] == len(DATA)
          and m["annotation"] == "boxes", m)
    fs = a.files(m)
    check("one archive file, streamed under the cap", a.fetch_file(fs[0], TMP / "w" / "a", max_bytes=10 ** 6)["bytes"]
          == len(DATA))


def test_mediatum(cfg):
    from weed_optimizer_framework.tools.collect import ChecksumMismatch
    print("record server and FTP")
    base = cfg.provider("mediatum")["base_url"]
    host = cfg.provider("mediatum")["ftp"]["host"]
    tray = b"tray zip bytes" * 10
    gt = b"track_id,label_id\n"
    sums = "%s  ./jpegs/POROL/1.zip\n%s  ./gt.csv\n%s  ./jpegs/CHEAL/2.zip\n" % (
        sha(tray, "sha512"), sha(gt, "sha512"), sha(b"x", "sha512"))
    srv = W.FakeFtp({"checksums.sha512": sums.encode(), "gt.csv": gt, "jpegs/POROL/1.zip": tray,
                     "jpegs/CHEAL/2.zip": b"x", "jpegs/ZZZZZ/3.zip": b"y"})
    net = W.make_net({("GET", base + "/services/export/node/42"): (200, {"nodelist": [[{"attributes": {
        "title": "Trays", "license": "by, http://creativecommons.org/licenses/by/4.0", "keywords": "a; b"}}]]})},
        ftp={host: srv})
    mt = prov(cfg, "mediatum", net)
    check("no search on the record server (the known items name its records)", mt.searchable is False
          and mt.search("x", 3) == [])
    m = mt.describe("42")
    check("describe reads the licence attribute", m["licence_text"].startswith("by, http") and m["title"] == "Trays")
    ki = cfg.known_item("mfwd_porol")
    from weed_optimizer_framework.tools.collect.fetch import _is_target_code_fn
    sel = dict(ki["select"], _is_target_code=_is_target_code_fn(cfg, None))
    fs = mt.files(m, sel)
    names = [f["name"] for f in fs]
    check("only the index file and the target code's directory are listed", names == ["gt.csv", "jpegs/POROL/1.zip"],
          names)
    check("each file carries its size and the published sha512", fs[1]["size"] == len(tray)
          and fs[1]["checksums"] == {"sha512": sha(tray, "sha512")} and fs[1]["group"] == "POROL")
    check("the FTP login is the known item's", srv.logins[-1] == ("m1717366", "m1717366"), srv.logins)
    r = mt.fetch_file(fs[1], TMP / "t" / "a", select=sel)
    check("the download is checked against sha512", r["sha512"] == sha(tray, "sha512") and r["bytes"] == len(tray))
    bad = dict(fs[1], checksums={"sha512": "0" * 128})
    check("a sha512 mismatch refuses", raises(lambda: mt.fetch_file(bad, TMP / "t" / "b", select=sel),
                                              ChecksumMismatch) is not None and not (TMP / "t" / "b").exists())


def test_transport():
    from weed_optimizer_framework.tools.collect import ByteCapExceeded
    from weed_optimizer_framework.tools.collect import transport as T
    print("transport")
    net = W.make_net({("GET", "https://h/big"): (200, DATA, {"Content-Length": str(len(DATA))}),
                      ("GET", "https://h/nolen"): (200, DATA)})
    e = raises(lambda: net.download("https://h/big", TMP / "tr" / "a", max_bytes=10, expected_size=len(DATA)),
               ByteCapExceeded)
    check("a listed size over the cap refuses before any byte", e is not None and not (TMP / "tr" / "a.tmp").exists())
    e = raises(lambda: net.download("https://h/nolen", TMP / "tr" / "b", max_bytes=100), ByteCapExceeded)
    check("a stream past the cap aborts and leaves nothing", e is not None and not (TMP / "tr" / "b").exists()
          and not (TMP / "tr" / "b.tmp").exists())
    n = T.Net({"base_timeout_s": 100, "min_rate_bytes_per_s": 1e6, "read_timeout_s": 5, "attempts": 1})
    import time
    d = n.deadline_for(5e9) - time.time()
    check("the deadline scales with size (100 s + 5 GB at 1 MB/s)", 5090 < d < 5110, d)
    check("URLs are redacted", "secret" not in T.redact("https://h/x?api_key=secret&q=1", {"token": "secret"}))
    signed = ("https://storage.googleapis.com/b/x.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=cred"
              "secret&X-Goog-Signature=sigsecret&Expires=1&Signature=s2secret&AWSAccessKeyId=akidsecret")
    check("a signed storage link (an export redirect) never reaches a record: its signature and credential are "
          "redacted", "secret" not in T.redact(signed) and "X-Goog-Algorithm" in T.redact(signed), T.redact(signed))
    (W.TMP / "home" / ".tok").write_text("filesecret\n")
    os.environ["SOME_ENV_TOKEN"] = "envsecret"
    try:
        check("a secret comes from the environment first, then a file",
              T.read_secret(["SOME_ENV_TOKEN"], ["~/.tok"])[0] == "envsecret"
              and T.read_secret(["NOPE_ENV"], ["~/.tok"]) == ("filesecret", "file:~/.tok"))
    finally:
        os.environ.pop("SOME_ENV_TOKEN", None)


def main():
    try:
        cfg = W.config()
        test_zenodo(cfg)
        test_mendeley(cfg)
        test_huggingface(cfg)
        test_kaggle(cfg)
        test_github(cfg)
        test_roboflow(cfg)
        test_weedai(cfg)
        test_mediatum(cfg)
        test_transport()
    finally:
        W.cleanup()
    print("\n%d failure(s)" % len(W.FAILURES))
    return 1 if W.FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
