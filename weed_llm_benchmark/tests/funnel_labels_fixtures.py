"""Synthetic worlds for the reference-labeller tests (test_funnel_sheets,
test_funnel_rl, test_funnel_qualify, test_funnel_recover), written in the
formats of docs/FUNNEL_AUDIT_RUNNER.md §4 and inc/verify.py.

setup(prefix) must run before the package is imported: it points INC_DIR and
REPO at a temporary tree and copies the contract and prereg_v1.json into it.
Everything else imports the package lazily.

sheet_world(tmp) builds a locked sample for the weed config's two boards:
  * B1-bound pool items (NDSU and an unaffiliated source) and independent
    photos (48 non-sentinel items: G1 12, G2 12, G2a 6, G3 8, G4 8, G0 2);
  * B2-bound items of the reference lab group (G1 2, G2 4, G2v 4) and two
    identity crops from a reference session no board exemplar uses (12);
  * sentinels: 2 reference crops and 1 copy (they fit B2 only), 8
    independent photos and 4 named non-target pool boxes (they fit both);
  * 12 guard pairs and 3 pair sentinels (sheets_v1_cluster only);
  * frames_v1.json with per-stratum predictions, sample_v1.csv, the key, and
    the sample-lock amendment in the temporary prereg.
FakeAdapter serves known truth, unit keys, the crop table and the pool.
"""
import json
import os
import pathlib
import shutil
import sys
import tempfile
import funnel_prereg as FPR  # noqa: E402

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
GIT_ROOT = PKG_ROOT.parent
LOCAL_INC = PKG_ROOT / "results" / "framework" / "inc"

ND = "project_agml__weed_crop_detection"
ND2 = "project_agml__greenhouse_crop_weed_detection"
LU = "rf_karthikeya-c8pvy__weed-detection-cwp10"
AN = "rf_tuf__weed-3434e"
COPY_SRC = "rf_agrobot-weed-workspace__weed-detection-sd89f"
REF = "train_core"


def setup(prefix):
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=prefix))
    os.environ["INC_DIR"] = str(tmp / "inc")
    os.environ["REPO"] = str(tmp / "repo")
    os.environ.pop("FUNNEL_CONTRACT", None)
    (tmp / "repo" / "docs").mkdir(parents=True)
    shutil.copy(GIT_ROOT / "docs" / "FUNNEL_AUDIT.md", tmp / "repo" / "docs" / "FUNNEL_AUDIT.md")
    fd = tmp / "inc" / "funnel"
    fd.mkdir(parents=True)
    FPR.write_pre_draw(fd / "prereg_v1.json", LOCAL_INC / "funnel" / "prereg_v1.json")
    # appended, not inserted: a package copy on PYTHONPATH (the mutation harness) comes first
    if str(PKG_ROOT) not in sys.path:
        sys.path.append(str(PKG_ROOT))
    return tmp


def have(*mods):
    """The names of the modules among mods that cannot be imported."""
    import importlib
    missing = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception:
            missing.append(m)
    return missing


def png(path, seed, w=64, h=48):
    import numpy as np
    from PIL import Image
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    Image.fromarray(a).save(path, format="PNG")
    return str(path)


class FakeAdapter(object):
    """The adapter functions the label modules call, over a synthetic world."""

    def __init__(self, known, keys, pool=None, base=None, labels=None, guard=None, features=None, probe=None):
        self._known = known
        self._keys = keys
        self._pool = pool or []
        self._base = base or []
        self._labels = labels or {}
        self._guard = guard
        self._features = features
        self._probe = probe

    def known_truth(self, domain, funnel_dir):
        return {k: [dict(it) for it in v] for k, v in self._known.items()}

    def unit_keys(self, unit_ids):
        return {u: dict(self._keys[u]) for u in unit_ids if u in self._keys}

    def crop_table(self):
        from weed_optimizer_framework.tools.inc import verify as V
        return V.Crops()

    def pool_rows(self, sources=None):
        return [dict(r) for r in self._pool if sources is None or r["source"] in sources]

    def base_rows(self):
        return [dict(r) for r in self._base]

    def label_rows(self, keys):
        return {k: list(self._labels[k]) for k in keys if k in self._labels}

    def never_train_guard(self):
        return self._guard

    def step1_features(self):
        return self._features, {"nshards": 1}

    def j1_scores(self, X):
        return self._probe(X)


def write_crops(rows):
    """step1/crops.csv (verify.CROP_FIELDS) and a crops_info.json; rows are
    dicts with set, key, image, source, group, box, cx, cy, w, h, W, H, label,
    src_name. Returns the rows with crop_id."""
    import csv
    from weed_optimizer_framework.tools.inc import verify as V
    V.STEP1.mkdir(parents=True, exist_ok=True)
    out = []
    with open(V.CROPS, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(V.CROP_FIELDS)
        for i, r in enumerate(rows):
            r = dict(r, crop_id=i)
            wr.writerow([r[f] if f not in ("cx", "cy", "w", "h") else "%.6f" % r[f] for f in V.CROP_FIELDS])
            out.append(r)
    return out


def write_manifest_rows(path, rows):
    from weed_optimizer_framework.tools.inc import common as C
    return C.write_manifest(path, rows)


def label_file(path, boxes):
    from weed_optimizer_framework.tools.inc import common as C
    C.write_yolo(path, boxes)
    return str(path), C.sha256_file(path)


def sheet_world(tmp, domain):
    """The locked sample of the module docstring. Returns a dict with the
    adapter, the sample rows, the key and the helpers the tests read."""
    from weed_optimizer_framework.tools.inc import common as C
    from weed_optimizer_framework.tools.inc import verify as V
    from weed_optimizer_framework.tools.funnel import draw as DR
    from weed_optimizer_framework.tools.funnel import domain as D
    from weed_optimizer_framework.tools.funnel import strata as ST
    from weed_optimizer_framework.tools.funnel import write_csv_atomic, write_json_atomic, write_jsonl_atomic
    fd = pathlib.Path(os.environ["INC_DIR"]) / "funnel"
    img_dir = tmp / "img"
    other = domain.other["id"]
    lab = domain.lab_of
    crops, known, keys = [], {k: [] for k in domain.kt_ids()}, {}
    seed = [100]

    def img(name, w=64, h=48):
        seed[0] += 1
        return png(img_dir / name, seed[0], w, h)

    # KT1: reference crops of classes 0-5, three sessions, four per class and session
    core_rows = []
    for cls in range(6):
        for s, sess in enumerate(("sA", "sB", "sC")):
            for j in range(4):
                key = "core_%d_%s_%d" % (cls, sess, j)
                im = img(key + ".png")
                core_rows.append({"key": key, "image": im, "session": sess, "cls": cls})
                crops.append({"set": "core", "key": key, "image": im, "source": REF, "group": sess, "box": 0,
                              "cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.6, "W": 64, "H": 48, "label": cls,
                              "src_name": domain.class_name(cls)})
    # copies (KT2) of two reference photographs
    copy_rows = []
    for j, cr in enumerate(core_rows[:2]):
        key = "copy_%d" % j
        im = img(key + ".png")
        copy_rows.append({"key": key, "image": im, "source": COPY_SRC, "train_core_key": cr["key"], "cls": cr["cls"]})
        crops.append({"set": "copy", "key": key, "image": im, "source": COPY_SRC, "group": COPY_SRC, "box": 0,
                      "cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.6, "W": 64, "H": 48, "label": other, "src_name": ""})
    # pool boxes
    pool_specs = []

    def pool(prefix, src, n, label, pred, group, stratum, kt=None, taxon=None, kind=None):
        for j in range(n):
            key = "%s_%s_%02d" % (prefix, src[:6], j)
            im = img(key + ".png")
            pool_specs.append({"key": key, "image": im, "source": src, "label": label, "pred": pred, "group": group,
                               "stratum": stratum, "kt": kt, "taxon": taxon, "kind": kind})
            crops.append({"set": "pool", "key": key, "image": im, "source": src, "group": src, "box": 0,
                          "cx": 0.45, "cy": 0.55, "w": 0.4, "h": 0.5, "W": 64, "H": 48, "label": label,
                          "src_name": "x"})
    pool("g1", AN, 6, other, 0, "G1", "G1/frame=noinfo/status=no_name/pred=Waterhemp")
    pool("g1", ND, 6, other, 8, "G1", "G1/frame=named/status=taxon_resolved/pred=PalmerAmaranth")
    pool("g2", ND, 12, 5, 12, "G2", "G2/source=%s/label=Ragweed/fail=argmax_other_confident" % ND)
    pool("g2a", ND2, 6, 0, 0, "G2a", "G2a/source=%s" % ND2)
    pool("g3", AN, 8, other, 0, "G3", "G3/unit=k:%s|*|0" % AN)
    pool("g4", AN, 8, other, other, "G4", "G4/frame=noinfo/argmax_target=no/band=1")
    pool("lu1", LU, 2, other, 0, "G1", "G1/frame=noinfo/status=numeric/pred=Waterhemp")
    pool("lu2", LU, 4, 0, 0, "G2", "G2/source=%s/label=Waterhemp/fail=cos_below_sigma" % LU)
    pool("luv", LU, 4, 0, 0, "G2v", "G2v/source=%s" % LU)
    pool("kt5", ND2, 4, other, 8, None, None, kt="KT5", taxon="Amaranthus retroflexus", kind="attractor")
    rows = write_crops(crops)
    crop_of = {(r["set"], r["key"]): r["crop_id"] for r in rows}
    # manifests the pair locator reads
    write_manifest_rows(C.manifest_path("train_core"),
                        [{"image": r["image"], "label": r["image"] + ".txt", "sha256": "s%d" % i, "label_sha256": "l",
                          "source": REF, "session": r["session"], "key": r["key"]} for i, r in enumerate(core_rows)])
    dev_rows = []
    for j in range(6):
        key = "dev_%02d" % j
        dev_rows.append({"image": img(key + ".png"), "label": "x", "sha256": "d%d" % j, "label_sha256": "l",
                         "source": "dev", "session": "", "key": key})
    write_manifest_rows(C.manifest_path("dev"), dev_rows)
    V.STEP1.mkdir(parents=True, exist_ok=True)
    with open(V.COPIES, "w") as fh:
        for r in copy_rows:
            fh.write(json.dumps({"key": r["key"], "image": r["image"], "source": r["source"],
                                 "train_core_key": r["train_core_key"]}) + "\n")
    pool_rows = [{"image": p["image"], "label": p["image"] + ".txt", "sha256": "p%d" % i, "label_sha256": "l",
                  "source": p["source"], "session": "", "key": p["key"]} for i, p in enumerate(pool_specs)]
    # a registry directory for the guard-pair source (G5 pairs name a relative path)
    reg_root = tmp / "repo" / "downloads" / "tuf"
    g5_rel = []
    for j in range(12):
        rel = "train/images/pair_%02d.png" % j
        png(reg_root / rel, 900 + j)
        g5_rel.append(rel)
    (tmp / "repo" / "results" / "framework").mkdir(parents=True, exist_ok=True)
    with open(V.REGISTRY, "w") as fh:
        json.dump({"datasets": {AN: {"local_path": str(reg_root), "class_names": []}}}, fh)
    # KT7 photos
    kt7_dir = fd / "kt7"
    kt7_rows = []
    targets7 = [0, 8, 5, 1]
    attr7 = ["Amaranthus retroflexus", "Bassia scoparia", "Ambrosia trifida"]
    n7 = 0
    for cls in targets7:
        for role, n in (("exemplar", 2), ("sentinel", 2), ("g0", 1)):
            for j in range(n):
                obs = 1000 + n7
                n7 += 1
                uid = "t7:%d/%d" % (obs, 1)
                kt7_rows.append({"id": uid, "truth": cls, "taxon": domain.target(cls)["taxon"], "kind": "target",
                                 "role": role, "file": "%d_1.png" % obs})
    for taxon in attr7:
        for role, n in (("exemplar", 2), ("sentinel", 1), ("g0", 1)):
            for j in range(n):
                obs = 1000 + n7
                n7 += 1
                uid = "t7:%d/%d" % (obs, 1)
                kt7_rows.append({"id": uid, "truth": other, "taxon": taxon, "kind": "attractor", "role": role,
                                 "file": "%d_1.png" % obs})
    import csv
    kt7_dir.mkdir(parents=True, exist_ok=True)
    with open(kt7_dir / "crops_kt7.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(V.CROP_FIELDS)
        for i, r in enumerate(kt7_rows):
            im = png(kt7_dir / "photos" / r["file"], 5000 + i, 40, 40)
            r["crop_id"] = i
            wr.writerow([i, "kt7", r["id"], im, "kt7", "kt7", 0, "0.500000", "0.500000", "1.000000", "1.000000",
                         40, 40, r["truth"], r["taxon"]])
    # known truth and unit keys
    for cr in core_rows:
        uid = "t1:%s#0" % cr["key"]
        it = {"id": uid, "kt": "KT1", "crop_id": crop_of[("core", cr["key"])], "crop_set": "core",
              "truth": cr["cls"], "truth_taxon": domain.target(cr["cls"])["taxon"], "truth_kind": "target",
              "claimed": False, "source": REF, "lab": lab(REF), "near_dup3": "n:%s" % cr["key"],
              "provenance": "prov:%s" % cr["key"], "session": cr["session"], "role": None}
        known["KT1"].append(it)
        keys[uid] = {k: it[k] for k in ("source", "lab", "near_dup3", "provenance")}
    for cp in copy_rows:
        uid = "t2:%s#0" % cp["key"]
        it = {"id": uid, "kt": "KT2", "crop_id": crop_of[("copy", cp["key"])], "crop_set": "copy",
              "truth": cp["cls"], "truth_taxon": domain.target(cp["cls"])["taxon"], "truth_kind": "target",
              "claimed": False, "source": COPY_SRC, "lab": lab(COPY_SRC), "near_dup3": "n:%s" % cp["key"],
              "provenance": "prov:%s" % cp["train_core_key"], "session": None, "role": None}
        known["KT2"].append(it)
        keys[uid] = {k: it[k] for k in ("source", "lab", "near_dup3", "provenance")}
    for r in kt7_rows:
        obs = r["id"].split(":")[1].split("/")[0]
        it = {"id": r["id"], "kt": "KT7", "crop_id": r["crop_id"], "crop_set": "kt7", "truth": r["truth"],
              "truth_taxon": r["taxon"], "truth_kind": r["kind"], "claimed": False, "source": "kt7",
              "lab": lab("kt7"), "near_dup3": "n:%s" % r["id"], "provenance": "prov:kt7|%s" % obs,
              "session": None, "role": r["role"]}
        known["KT7"].append(it)
        keys[r["id"]] = {k: it[k] for k in ("source", "lab", "near_dup3", "provenance")}
    for p in pool_specs:
        uid = "b:%s#0" % p["key"]
        keys[uid] = {"source": p["source"], "lab": lab(p["source"]), "near_dup3": "n:%s" % p["key"],
                     "provenance": "prov:%s" % p["key"]}
        if p["kt"] == "KT5":
            known["KT5"].append(dict(keys[uid], id=uid, kt="KT5", crop_id=crop_of[("pool", p["key"])],
                                     crop_set="pool", truth=other, truth_taxon=p["taxon"], truth_kind="attractor",
                                     claimed=True, session=None, role=None))
    # the sample: items by group
    sample, key_rows = [], []

    def add(uid, group, stratum, unit="box", sheet_class="pool", truth=None, taxon=None, kind=None, pair=None,
            kt="", crop_id=""):
        iid = DR.item_id(uid, group)
        k = keys.get(uid, {"source": uid[2:].split("|")[0] if uid.startswith("p:") else "", "lab": "",
                           "near_dup3": "", "provenance": ""})
        row = {"item_id": iid, "unit_id": uid, "unit": unit, "group": group, "stratum": stratum,
               "source": k["source"], "image_key": "", "crop_id": crop_id, "pi": "0.5", "pi_parts": "{}",
               "seed_text": "funnel/v1/%s" % stratum, "draw_rank": len(sample), "sheet_class": sheet_class,
               "lab": k["lab"] or (lab(k["source"]) if k["source"] else ""), "near_dup3": k["near_dup3"],
               "provenance": k["provenance"], "kt": kt}
        if group == "G0":
            row.update({"unit_id": "G0:%s" % iid, "unit": "", "stratum": "G0/all", "source": "", "lab": "",
                        "near_dup3": "", "provenance": "", "crop_id": ""})
        sample.append(row)
        key_rows.append({"item_id": iid, "unit_id": uid, "truth": truth, "truth_taxon": taxon, "truth_kind": kind,
                         "pair_truth": pair})
        return iid
    for p in pool_specs:
        if p["group"]:
            add("b:%s#0" % p["key"], p["group"], p["stratum"], crop_id=crop_of[("pool", p["key"])])
    g0 = [r for r in kt7_rows if r["role"] == "g0"][:2]
    for r in g0:
        add(r["id"], "G0", "G0/all", truth=r["truth"], taxon=r["taxon"], kind=r["kind"])
    ex_sessions = ST.exemplar_sessions(domain, known)
    used = set()
    for b in ex_sessions.values():
        for n, ss in b.items():
            used.update(ss)
    rag = domain.class_id("Ragweed")
    free = [it for it in known["KT1"] if it["truth"] == rag and it["session"] not in used] or \
        [it for it in known["KT1"] if it["truth"] == rag]
    ident = free[:2]
    for it in ident:
        add(it["id"], "identity", "identity/class=Ragweed", truth=it["truth"], taxon=it["truth_taxon"],
            kind="target", kt="KT1", crop_id=it["crop_id"])
    other_kt1 = [it for it in known["KT1"] if it not in ident and it["truth"] != rag]
    sent = [(it, "KT1") for it in other_kt1[:2]] + [(known["KT2"][0], "KT2")]
    sent += [(it, "KT7") for it in known["KT7"] if it["role"] == "sentinel"][:8]
    sent += [(it, "KT5") for it in known["KT5"]]
    for it, kt in sent:
        add(it["id"], "sentinel", "sentinel/kt=%s/truth_kind=%s" % (kt, it["truth_kind"]), truth=it["truth"],
            taxon=it["truth_taxon"], kind=it["truth_kind"], kt=kt, crop_id=it["crop_id"])
    for j, rel in enumerate(g5_rel):
        add("p:%s|%s|dev|%s" % (AN, rel, dev_rows[j % 6]["key"]), "G5", "G5/source=%s/split=dev/bits=0-2" % AN,
            unit="pair", sheet_class="eval")
    add("p:%s|%s|%s|%s" % (COPY_SRC, copy_rows[0]["key"], REF, copy_rows[0]["train_core_key"]), "pair_sentinel",
        "pair_sentinel/kind=positive", unit="pair", sheet_class="eval", pair="same", kt="KT2")
    add("p:%s|%s|%s|%s" % (COPY_SRC, copy_rows[1]["key"], REF, copy_rows[1]["train_core_key"]), "pair_sentinel",
        "pair_sentinel/kind=positive", unit="pair", sheet_class="eval", pair="same", kt="KT2")
    neg_b = next(p for p in pool_specs if p["source"] == AN)["key"]
    add("p:calibration:%s|%s|%s|%s|%s" % (REF, AN, neg_b, REF, core_rows[5]["key"]), "pair_sentinel",
        "pair_sentinel/kind=negative", unit="pair", sheet_class="eval", pair="different")
    # frames: every sampled unit plus extra rows that set the strata's prediction shares
    frame_rows = {}
    for p in pool_specs:
        if not p["group"]:
            continue
        uid = "b:%s#0" % p["key"]
        frame_rows.setdefault(p["group"], []).append(
            {"unit_id": uid, "unit": "box", "stratum": p["stratum"], "source": p["source"], "image_key": p["key"],
             "crop_id": crop_of[("pool", p["key"])], "lab": lab(p["source"]), "near_dup3": keys[uid]["near_dup3"],
             "provenance": keys[uid]["provenance"], "label": p["label"], "pred": p["pred"], "score": "",
             "allowed_judges": "", "extra": "{}"})
    groups = {}
    fdir = fd / "frames_v1"
    for g, rws in sorted(frame_rows.items()):
        sha = write_csv_atomic(fdir / ("%s.csv" % g), ST.FRAME_FIELDS, rws)
        groups[g] = {"design": "srs", "N": len(rws), "planned": len(rws), "minimum": None,
                     "file": {"path": str(fdir / ("%s.csv" % g)), "sha256": sha}, "strata": {}}
    frames_sha = write_json_atomic(fd / "frames_v1.json", {"format": "funnel-frames/1", "groups": groups})
    sample_sha = write_csv_atomic(fd / "sample_v1.csv", DR.SAMPLE_FIELDS, sample)
    planted = {"share": 0.5, "n_pos": 1, "n_neg": 1, "seed_text": "funnel/v1/G0/share"}
    key_sha = write_jsonl_atomic(fd / "sample_v1_key.jsonl", sorted(key_rows, key=lambda r: r["item_id"])
                                 + [{"planted": planted}])
    pre = D.load_prereg(fd / "prereg_v1.json")
    D.append_amendment(fd / "prereg_v1.json", {
        "id": D.next_amendment_id(pre), "kind": "sample_lock", "date": "2026-09-28",
        "prereg_core_sha256": pre.core_sha256,
        "sample_sha256": sample_sha, "key_sha256": key_sha, "frames_sha256": frames_sha,
        "name_status_v2_sha256": "0" * 64, "frame_sizes": {g: v["N"] for g, v in groups.items()},
        "confirmatory_frames": {"H2a": 1, "H2b": 1, "H4_named": 1, "H4_noinfo": 1}})
    adapter = FakeAdapter(known, keys, pool=pool_rows)
    return {"adapter": adapter, "sample": sample, "key": {r["item_id"]: r for r in key_rows}, "known": known,
            "keys": keys, "funnel_dir": fd, "pool_specs": pool_specs, "crops": rows, "kt7": kt7_rows,
            "identity": ident, "sentinels": sent, "core": core_rows, "copies": copy_rows}


# ------------------------------------------------------------ recovery world
ND2 = "project_agml__greenhouse_crop_weed_detection"
MH = "project_agml__mh_weed16_weed_detection"
XS = "rf_test-gzc3r__crop-mfete"
P2 = "rf_srec-dthh0__crop-weed-poxtn"
QS = "rf_test-8qezo__weed-detection-ycai2"
NL = "rf_school-5ult5__weed-6a90d"
NR = "fvossel__csgo_player_detection"


def test_domain(tmp):
    """The weed config with licences for the synthetic recovery sources (and
    none for NL), written to tmp/weed_test.json and loaded."""
    from weed_optimizer_framework.tools.funnel import domain as D
    raw = json.loads((PKG_ROOT / "weed_optimizer_framework" / "tools" / "funnel" / "domains" /
                      "weed.json").read_text())
    lic = raw["sources"].setdefault("licences", {})
    for s in (AN, P2, XS, MH, ND, ND2, QS, NR):       # NR: licensed, so only not_recoverable excludes it
        lic.setdefault(s, "CC BY 4.0 (synthetic test licence)")
    lic.pop(NL, None)
    p = tmp / "weed_test.json"
    p.write_text(json.dumps(raw, indent=1))
    return D.load(p)


def recover_world(tmp, domain):
    """Step 1 files, the census ledger, frames, audit, qualification,
    relation, class maps, leak record, name status, gold and a judge score
    file for the recovery tests. Returns a dict of planted keys and helpers."""
    import numpy as np
    from weed_optimizer_framework.tools.inc import common as C
    from weed_optimizer_framework.tools.inc import verify as V
    from weed_optimizer_framework.tools.funnel import qualify as Q
    from weed_optimizer_framework.tools.funnel import strata as ST
    from weed_optimizer_framework.tools.funnel import write_csv_atomic, write_json_atomic, json_text
    fd = pathlib.Path(os.environ["INC_DIR"]) / "funnel"
    O = domain.other["id"]
    W, P, R, S = (domain.class_id(n) for n in ("Waterhemp", "PalmerAmaranth", "Ragweed", "Sicklepod"))
    lab = domain.lab_of
    images, ledger, frames = [], [], {}
    seed = [7000]
    cid = [0]

    def box_row(key, src, b, label=None, src_id=None, status=None, s8=None, s9=None, s10=None, pred=None, group=None,
                stratum=None, nd=None, small=False):
        uid = "b:%s#%d" % (key, b)
        c = None
        if not small:
            c = cid[0]
            cid[0] += 1
        ledger.append({"id": uid, "unit": "box", "source": src, "key": key, "box": b, "crop_id": c, "label": label,
                       "src_id": str(src_id), "src_name": "", "name_status_v2": status, "lab": lab(src),
                       "near_dup3": nd or "n:%s" % key, "provenance": "prov:%s" % key, "pred": pred,
                       "path": {"S4": "pass", "S5": "pass", "S6": "small" if small else "embedded",
                                "S8": s8, "S9": s9, "S10": s10}, "kt": []})
        if group:
            frames.setdefault(group, []).append({"unit_id": uid, "unit": "box", "stratum": stratum, "source": src,
                                                 "image_key": key, "crop_id": "" if c is None else c, "lab": lab(src),
                                                 "near_dup3": nd or "n:%s" % key, "provenance": "prov:%s" % key,
                                                 "label": label, "pred": "" if pred is None else pred, "score": "",
                                                 "allowed_judges": "", "extra": "{}"})
        return c

    def image(key, src, boxes, nd=None):
        """boxes: [(cls, cx, cy, w, h, dict of box_row kwargs)]"""
        seed[0] += 1
        path = png(tmp / "pool" / src[:10] / ("%s.png" % key), seed[0], 64, 48)
        text = V._yolo_text([b[:5] for b in boxes])
        lp, lsha = V._label_file(V.LABELS_DIR, src, key, text)
        V._write_label(lp, text)
        crops = []
        for i, b in enumerate(boxes):
            crops.append(box_row(key, src, i, nd=nd, **b[5]))
        images.append({"key": key, "source": src, "image": path, "label": str(lp), "label_sha256": lsha,
                       "sha256": C.sha256_file(path), "boxes": [list(b[:5]) for b in boxes], "nd": nd, "crops": crops})
        return crops
    ok = "G2v/source=%s" % ND2
    s_rag = "G2/source=%s/label=Ragweed/fail=p_below_tau" % ND
    s_wh_bad = "G2/source=%s/label=Waterhemp/fail=argmax_other_confident" % ND
    s_wh = "G2/source=%s/label=Waterhemp/fail=p_below_tau" % ND
    s_g1 = "G1/frame=noinfo/status=no_name/pred=Waterhemp"
    s_g1_named = "G1/frame=named/status=target_related/pred=Waterhemp"
    s_g4 = "G4/frame=noinfo/argmax_target=no/band=1"
    s_g4y = "G4/frame=noinfo/argmax_target=yes/band=3"
    s_mh = "G3/unit=c:%s|12" % MH
    s_xs = "G3/unit=c:%s|0" % XS
    t = dict  # shorthand
    # R-V: a verified target box vetoed by an other-class conflict (and the near-eval twin)
    image("nd2_veto", ND2, [(W, .3, .3, .2, .2, t(label=W, src_id=0, status="target", s8="verified", s9="n/a",
                                                     s10="conflict", pred=W, group="G2v", stratum=ok)),
                            (O, .7, .7, .2, .2, t(label=O, src_id=3, status="generic", s8="n/a", s9="conflict",
                                                     s10="conflict", pred=P, group="G1", stratum=s_g1)),
                            (O, .5, .8, .1, .1, t(label=O, src_id=3, status="generic", s8="n/a", s9="other_ok",
                                                     s10="conflict", pred=O))])
    image("nd2_near", ND2, [(W, .2, .2, .1, .1, t(label=W, src_id=0, status="target", s8="verified", s9="n/a",
                                                     s10="conflict", pred=W, group="G2v", stratum=ok)),
                            (O, .6, .55, .8, .9, t(label=O, src_id=3, status="generic", s8="n/a", s9="conflict",
                                                      s10="conflict", pred=P))])
    # R-A
    image("nd_auth1", ND, [(R, .3, .3, .2, .2, t(label=R, src_id=5, status="target", s8="unknown", s9="n/a",
                                                    s10="unknown", pred=O, group="G2", stratum=s_rag)),
                           (O, .7, .3, .2, .2, t(label=O, src_id=7, status="taxon_resolved", s8="n/a", s9="conflict",
                                                    s10="unknown", pred=P, group="G1", stratum=s_g1_named)),
                           (O, .5, .7, .2, .2, t(label=O, src_id=9, status="generic", s8="n/a", s9="conflict",
                                                    s10="unknown", pred=W))])
    image("nd_auth2", ND, [(W, .5, .5, .3, .3, t(label=W, src_id=0, status="target", s8="conflict", s9="n/a",
                                                    s10="conflict", pred=O, group="G2", stratum=s_wh_bad))])
    image("nd_auth3", ND, [(W, .5, .5, .3, .3, t(label=W, src_id=0, status="target", s8="unknown", s9="n/a",
                                                    s10="unknown", pred=O, group="G2", stratum=s_wh))])
    image("nd_auth4", ND, [(W, .5, .5, .3, .3, t(label=W, src_id=0, status="target", s8="unknown", s9="n/a",
                                                    s10="unknown", pred=O, group="G2", stratum=s_wh))])
    # R-C and R-T
    image("mh_class1", MH, [(O, .3, .3, .2, .2, t(label=O, src_id=12, status="numeric", s8="n/a", s9="other_ok",
                                                     s10="admitted", pred=O, group="G3", stratum=s_mh)),
                            (O, .7, .7, .2, .2, t(label=O, src_id=8, status="numeric", s8="n/a", s9="conflict",
                                                     s10="conflict", pred=domain.class_id("SpottedSpurge")))])
    image("xs_syn1", XS, [(O, .5, .5, .3, .3, t(label=O, src_id=0, status="target_synonym", s8="n/a",
                                                   s9="other_ok", s10="admitted", pred=O, group="G3", stratum=s_xs))])
    # R-J
    judge = {}

    def j_image(key, src, nd=None, extra=None):
        boxes = [(O, .3, .4, .25, .25, t(label=O, src_id=0, status="no_name", s8="n/a", s9="conflict",
                                         s10="conflict", pred=W, group="G1", stratum=s_g1))] + list(extra or [])
        cs = image(key, src, boxes, nd=nd)
        judge[cs[0]] = "Waterhemp"
        return cs
    cs = j_image("p2_j1", P2, extra=[(O, .7, .7, .2, .2, t(label=O, src_id=0, status="no_name", s8="n/a",
                                                               s9="other_ok", s10="conflict", pred=O, group="G4",
                                                               stratum=s_g4))])
    judge[cs[1]] = "PalmerAmaranth"
    cs = image("p2_rel", P2, [(O, .3, .3, .2, .2, t(label=O, src_id=4, status="target_related", s8="n/a",
                                                       s9="conflict", s10="conflict", pred=W, group="G1",
                                                       stratum=s_g1_named)),
                              (O, .7, .7, .2, .2, t(label=O, src_id=0, status="no_name", s8="n/a", s9="conflict",
                                                       s10="conflict", pred=W, group="G1", stratum=s_g1))])
    judge[cs[0]] = "Waterhemp"
    judge[cs[1]] = "Waterhemp"
    # R-J: the judge's target is not the step-1 probe's predicted class the gate measured
    cs = image("p2_jp", P2, [(O, .4, .4, .2, .2, t(label=O, src_id=0, status="no_name", s8="n/a", s9="conflict",
                                                   s10="conflict", pred=W, group="G1", stratum=s_g1))])
    judge[cs[0]] = "PalmerAmaranth"
    # R-J: a relabelled box plus an other_ok box only an unqualified judge calls a target (masked)
    cs = image("p2_mask", P2, [(O, .3, .3, .2, .2, t(label=O, src_id=0, status="no_name", s8="n/a", s9="conflict",
                                                     s10="conflict", pred=W, group="G1", stratum=s_g1)),
                               (O, .7, .7, .2, .2, t(label=O, src_id=0, status="no_name", s8="n/a", s9="other_ok",
                                                     s10="conflict", pred=O, group="G4", stratum=s_g4))])
    judge[cs[0]] = "Waterhemp"
    unqualified = {cs[1]: "Goosegrass"}
    # R-J in an H4 stratum: an other_ok box whose probe argmax is a target
    cs = image("p2_g4y", P2, [(O, .5, .5, .3, .3, t(label=O, src_id=0, status="no_name", s8="n/a", s9="other_ok",
                                                    s10="admitted", pred=W, group="G4", stratum=s_g4y))])
    judge[cs[0]] = "Waterhemp"
    for g in range(11):
        for j in range(4):
            j_image("an_j_%02d_%d" % (g, j), AN, nd="n:an_group_%02d" % g)
    j_image("qs_j", QS)
    image("qs_veto", QS, [(W, .3, .3, .2, .2, t(label=W, src_id=0, status="target", s8="verified", s9="n/a",
                                                   s10="conflict", pred=W)),
                          (O, .7, .7, .2, .2, t(label=O, src_id=3, status="generic", s8="n/a", s9="conflict",
                                                   s10="conflict", pred=P))])
    j_image("nl_j", NL)
    j_image("nr_j", NR)
    j_image("p2_base", P2)
    # Step 1 files
    V.STEP1.mkdir(parents=True, exist_ok=True)
    pool = [{"image": im["image"], "label": im["label"], "sha256": im["sha256"], "label_sha256": im["label_sha256"],
             "source": im["source"], "session": "", "key": im["key"]} for im in images]
    C.write_manifest(V.POOL, pool)
    near_hash = 0x0F0F0F0F0F0F0F0F
    with open(V.POOL_META, "w") as fh:
        for im in sorted(images, key=lambda x: x["key"]):
            h = near_hash if im["key"] == "nd2_near" else C.stable_int("dh/" + im["key"], 2 ** 62)
            fh.write(json.dumps({"key": im["key"], "source": im["source"], "W": 64, "H": 48, "dhash": h,
                                 "boxes": im["boxes"], "src": [[0, ""] for _ in im["boxes"]]}) + "\n")
    base = [r for r in pool if r["key"] == "p2_base"]
    # frames
    groups = {}
    for g, rws in sorted(frames.items()):
        p = fd / "frames_v1" / ("%s.csv" % g)
        sha = write_csv_atomic(p, ST.FRAME_FIELDS, rws)
        groups[g] = {"file": {"path": str(p), "sha256": sha}, "N": len(rws)}
    write_json_atomic(fd / "frames_v1.json", {"format": "funnel-frames/1", "groups": groups})
    with open(fd / "ledger.jsonl", "w") as fh:
        for r in sorted(ledger, key=lambda r: r["id"]):
            fh.write(json.dumps(r) + "\n")

    def stratum(group, sid, lb=0.9, point=0.95, event="label"):
        # the event each gate reads (recover.BOX_EVENT, CLASS_EVENT, CLASS_BOX_EVENT; estimate.py's vocabulary)
        return {"group": group, "stratum": sid, "N": 10, "n": 10, "n_labelled": 10, "event": event,
                "labeller": "RL-B", "level": "species", "estimate": point, "interval": [lb, 0.99],
                "method": "wilson", "rogan_gladen": {"applied": True, "se": None, "sp": None, "flag": None},
                "unsure": {"as_no": {"estimate": point, "interval": [lb, 0.99]},
                           "as_yes": {"estimate": point, "interval": [lb, 0.99]}}}
    audit = {"format": "funnel-audit/1", "valid": True, "stop": None, "calibration_overlap": [], "inputs": {},
             "strata": [stratum("G2v", ok), stratum("G2", s_rag), stratum("G2", s_wh_bad, lb=0.7, point=0.8),
                        stratum("G2", s_wh), stratum("G1", s_g1, event="pred"),
                        stratum("G1", s_g1_named, event="pred"),
                        stratum("G4", s_g4, lb=0.02, point=0.05, event="pred"), stratum("G4", s_g4y, event="pred"),
                        stratum("G3", s_mh, event="purity:Sicklepod"), stratum("G3", s_mh, event="label:Sicklepod"),
                        stratum("G3", s_xs, event="purity:Waterhemp")],
             "hypotheses": {h: {"verdict": "supported"} for h in ("H1", "H2a", "H3a", "H7")}}
    write_json_atomic(fd / "audit_v1.json", audit)
    write_json_atomic(fd / "rl_qualification.json", {"format": "funnel-rl-qualification/1",
                                                     "identity": {"Ragweed": {"pass": True, "before": ["R-A"]}}})
    write_json_atomic(fd / "relation_geometry_v1.json", {"h1_pre": {ND: {"pass": True, "why": "ids agree"}}})
    write_json_atomic(fd / "class_maps.json", {"proposals": [
        {"source": MH, "src_id": "12", "src_name": "12", "map_to": "Sicklepod", "via": "card+geometry",
         "status": "proposed", "reason": "card id 12 and geometry agree"},
        {"source": MH, "src_id": "5", "src_name": "5", "map_to": "MorningGlory", "via": "card",
         "status": "proposed", "reason": "card only"}]})
    # the copy detector's two readings (contract §14 A2): version 1 quarantined XS too (the per-pair calibration
    # read per source); version 2, the one the prereg's amendment requires, quarantines QS only
    write_json_atomic(fd / "leak_v1.json", {"format": "funnel-leak/1", "h6a": {"quarantine": [QS, XS]},
                                            "h6b": {"base_copy": True, "increment_copies": {"S1": 3},
                                                    "incident": True},
                                            "h6c": {"groups": {"NDSU": [ND, ND2]}}, "calibration": {"ok": True}})
    write_json_atomic(fd / "leak_v2.json", {"format": "funnel-leak/2", "detector_version": 2,
                                            "h6a": {"quarantine": [QS]},
                                            "h6b": {"base_copy": False, "increment_flagged": {}, "incident": False},
                                            "h6c": {"groups": {"NDSU": [ND, ND2]}}, "calibration": {"ok": True}})
    write_json_atomic(fd / "name_status_v2.json", {"names": [
        {"source": XS, "src_id": "0", "name": "Amaranthus rudis", "status_v2": "target_synonym", "via": "scientific",
         "taxon": domain.target(W)["taxon"]},
        {"source": ND, "src_id": "7", "name": "redroot pigweed", "status_v2": "taxon_resolved", "via": "vernacular",
         "taxon": "Amaranthus retroflexus"},
        {"source": P2, "src_id": "4", "name": "pigweed sp", "status_v2": "target_related", "via": "pattern",
         "taxon": None}]})
    write_csv_atomic(fd / "gold_v1.csv", ("item_id", "unit_id", "answer", "answer_level", "answer_taxon"),
                     [{"item_id": "x", "unit_id": "b:nd_auth3#0", "answer": "other", "answer_level": "genus",
                       "answer_taxon": "Amaranthus"}])
    # the judge: qualification entry, material file, score file
    (fd / "judges").mkdir(parents=True, exist_ok=True)
    mat_doc = {"format": "funnel-judge-material/1",
               "judges": {"J-knn2": {"all": {"source": [], "lab": [], "near_dup3": [], "provenance": []}},
                          "J-knn1": {"all": {"source": [], "lab": [], "near_dup3": [], "provenance": []}}}}
    mpath = fd / Q.MATERIAL_FILE
    mpath.write_text(json_text(mat_doc))
    msha = C.sha256_file(mpath)
    empty = Q._hash_list([])
    write_json_atomic(fd / Q.JUDGE_FILE, {
        "format": "funnel-judge-qualification/1", "material": {"path": str(mpath), "sha256": msha},
        "judges": {"J-knn2": {"kind": "knn", "by_type": {"other_noinfo": {"qualified": True},
                                                         "other_named": {"qualified": True},
                                                         "shifted_target": {"qualified": True}},
                              "by_lab_scope": {},
                              "calibration_material": {"kt": [], "sources": [], "labs": [], "near_dup3": empty,
                                                       "provenance": empty,
                                                       "lists": {"path": str(mpath), "sha256": msha,
                                                                 "judge": "J-knn2", "scope": "all"}}},
                   "J-knn1": {"kind": "knn", "by_type": {t_: {"qualified": False} for t_ in Q.TYPES},
                              "by_lab_scope": {},
                              "calibration_material": {"kt": [], "sources": [], "labs": [], "near_dup3": empty,
                                                       "provenance": empty,
                                                       "lists": {"path": str(mpath), "sha256": msha,
                                                                 "judge": "J-knn1", "scope": "all"}}}}})
    labels = list(domain.target_names) + ["other"]
    idx = np.array(sorted(judge), dtype=np.int64)
    top = np.array([labels.index(judge[i]) for i in idx], dtype=np.int16)
    np.savez(fd / "judges" / "J-knn2__crops.npz", unit_index=idx, P=np.zeros((len(idx), len(labels)), np.float16),
             top=top, meta=np.array(json.dumps({"judge": "J-knn2", "labels": labels, "set": "crops"})))
    # an unqualified judge that can answer a non-target: its target calls mask (recover.mask_judges_of);
    # a targets-only label space (a kNN bank of target crops) would not
    labels1 = list(domain.target_names) + ["other"]
    idx1 = np.array(sorted(unqualified), dtype=np.int64)
    top1 = np.array([labels1.index(unqualified[i]) for i in idx1], dtype=np.int16)
    np.savez(fd / "judges" / "J-knn1__crops.npz", unit_index=idx1, P=np.zeros((len(idx1), len(labels1)), np.float16),
             top=top1, meta=np.array(json.dumps({"judge": "J-knn1", "labels": labels1, "set": "crops"})))
    guard_hit = C.NeverTrainGuard([(near_hash, "dev", "dev_0001")])
    guard_clean = C.NeverTrainGuard([(0x1234567812345678, "dev", "dev_0001")])
    labels_by_key = {im["key"]: [tuple(b) for b in im["boxes"]] for im in images}
    adapter = FakeAdapter({}, {}, pool=pool, base=base, labels=labels_by_key, guard=guard_hit)
    return {"adapter": adapter, "images": {im["key"]: im for im in images}, "pool": pool, "base": base,
            "guard_hit": guard_hit, "guard_clean": guard_clean, "near_hash": near_hash, "funnel_dir": fd,
            "strata": {"veto": ok, "rag": s_rag, "wh_bad": s_wh_bad, "wh": s_wh, "g1": s_g1, "g4": s_g4,
                       "g4y": s_g4y, "mh": s_mh, "xs": s_xs}, "classes": {"W": W, "P": P, "R": R, "S": S, "O": O},
            "judge": judge, "unqualified": unqualified}
