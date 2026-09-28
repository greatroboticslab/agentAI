"""Test support for the funnel statistics modules (test_funnel_init, _domain,
_ledger, _claims, _strata, _draw, _estimate).

setup(prefix) must run before the package is imported: it points INC_DIR and
REPO at a temporary tree and copies docs/FUNNEL_AUDIT.md and prereg_v1.json
into it (runner §1.5).

build_world(fd, ...) writes a small census in the formats of runner §4.1-4.4
(census_v1.json, ledger.jsonl, name_status_v2.json, funnel_ledger.json,
guard_pairs_v1.csv, leak_pairs_v1.csv) for the weed config, and returns a
FakeAdapter serving the known truth of runner §4.6. Every source slug and
class name comes from domains/weed.json; every count is chosen here and is
synthetic.
"""
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile

PKG_ROOT = pathlib.Path(__file__).resolve().parents[1]
GIT_ROOT = PKG_ROOT.parent
LOCAL_INC = PKG_ROOT / "results" / "framework" / "inc"

FAILURES = []
SKIPS = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def skip(name, reason):
    print("  SKIP: %s (%s)" % (name, reason))
    SKIPS.append(name)


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


def finish():
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(FAILURES + SKIPS) or "none"))
    sys.exit(1 if FAILURES else 0)


def setup(prefix):
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=prefix))
    os.environ["INC_DIR"] = str(tmp / "inc")
    os.environ["REPO"] = str(tmp / "repo")
    os.environ.pop("FUNNEL_CONTRACT", None)
    os.environ.pop("FUNNEL_DOMAINS_DIR", None)
    (tmp / "repo" / "docs").mkdir(parents=True)
    shutil.copy(GIT_ROOT / "docs" / "FUNNEL_AUDIT.md", tmp / "repo" / "docs" / "FUNNEL_AUDIT.md")
    fd = tmp / "inc" / "funnel"
    fd.mkdir(parents=True)
    shutil.copy(LOCAL_INC / "funnel" / "prereg_v1.json", fd / "prereg_v1.json")
    sys.path.insert(0, str(PKG_ROOT))
    return tmp


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- the world
ND = "project_agml__weed_crop_detection"          # authoritative, NDSU group
ND2 = "project_agml__greenhouse_crop_weed_detection"
LU = "rf_karthikeya-c8pvy__weed-detection-cwp10"  # reference lab group
AN = "rf_tuf__weed-3434e"                         # anonymous (no-name) source
MH = "project_agml__mh_weed16_weed_detection"     # numeric, card-resolved (KT6)
PER = "rf_university-of-peradeniya__weed-detection-ekr8i"
COPY_SRC = "rf_agrobot-weed-workspace__weed-detection-sd89f"
REF = "train_core"


class FakeAdapter(object):
    """known_truth and crop_table of the adapter interface. The crop table
    labels the reference copies' crops with their join label: two in three
    KT2 copy boxes are hidden targets (joined to the other class)."""

    def __init__(self, known, other_id=None):
        self.known = known
        if other_id is None:      # the other class is the truth of the claimed named non-targets (KT5)
            other_id = [it["truth"] for it in known.get("KT5", []) if it.get("truth") is not None][0]
        self.other_id = int(other_id)

    def known_truth(self, domain, funnel_dir):
        return self.known

    def crop_table(self):
        import types
        import numpy as np
        other = self.other_id
        ids = [int(it["crop_id"]) for items in self.known.values() for it in items if it.get("crop_id") is not None]
        label = np.full(max(ids) + 1, -1, dtype=np.int64)
        for items in self.known.values():
            for it in items:
                if it.get("crop_id") is not None and it.get("truth") is not None:
                    label[int(it["crop_id"])] = int(it["truth"])
        for it in self.known.get("KT2", []):
            if hidden_copy(it):
                label[int(it["crop_id"])] = other
        return types.SimpleNamespace(label=label, path=None, sha=sha(json.dumps(label.tolist())))


def hidden_copy(item):
    """The world's hidden KT2 targets: two in three copy boxes."""
    return int(item["crop_id"]) % 3 != 0


def _box(dom, key, b, source, label, src_id, src_name, status, *, size="embedded", tc="n/a", oc="n/a",
         ir="admitted", pred=None, p=None, cos=None, ptm=0.1, fail=None, crop_id=None, nd3=None, kt=None,
         ev="evidenced"):
    lab = dom.lab_of(source)
    path = {"S3": "kept", "S4": "pass", "S5": "pass", "S6": size, "S7": "target" if dom.is_target(label) else "other",
            "S8": tc, "S9": oc, "S10": ir, "S11": "base", "S12": ev}
    failed = []
    if size == "small":
        failed.append("S6")
    if tc in ("conflict", "unknown"):
        failed.append("S8")
    if oc == "conflict":
        failed.append("S9")
    if ir != "admitted":
        failed.append("S10")
    return {"id": "b:%s#%d" % (key, b), "unit": "box", "source": source, "key": key, "box": b,
            "crop_id": crop_id, "label": label, "src_id": str(src_id), "src_name": src_name, "wh_px": [40, 40],
            "name_status_v1": "x", "name_status_v2": status, "lab": lab,
            "near_dup3": nd3 or ("n:%s" % key), "provenance": "prov:%s" % key,
            "pred": pred, "p": p, "cos": cos, "p_target_max": ptm if size == "embedded" else None,
            "p_other": None, "second": None, "p_second": None, "fail": fail, "blockers": None, "path": path,
            "failed_stages": failed, "first_cause": failed[0] if failed else None,
            "sole_cause": failed[0] if len(failed) == 1 else None, "kt": list(kt or [])}


def world_rows(dom, n_scale=1):
    """The ledger box rows of the synthetic world (sorted by id)."""
    import numpy as np
    rows = []
    other = dom.other["id"]
    rag, wh, pa = dom.class_id("Ragweed"), dom.class_id("Waterhemp"), dom.class_id("PalmerAmaranth")
    mg, sp = dom.class_id("MorningGlory"), dom.class_id("Sicklepod")
    cid = [0]

    def nxt():
        cid[0] += 1
        return cid[0] - 1
    r = np.random.default_rng(7)
    # NDSU authoritative source: Ragweed unknown / conflict (G2), Waterhemp verified (G2a) some vetoed (G2v)
    for i in range(40 * n_scale):
        key = "nd_img%03d" % i
        rows.append(_box(dom, key, 0, ND, rag, 8, "Ragweed", "target", tc="unknown" if i % 2 else "conflict",
                         ir="unknown", pred=other if i % 2 else wh, p=0.4, cos=0.5, fail="p_below_tau",
                         crop_id=nxt(), kt=["KT4"]))
        rows.append(_box(dom, key, 1, ND, wh, 9, "Waterhemp", "target", tc="verified",
                         ir="unknown" if i < 10 else "admitted", pred=wh, p=0.9, cos=0.9, crop_id=nxt(), kt=["KT4"]))
        rows.append(_box(dom, key, 2, ND, other, 5, "Redroot pigweed", "taxon_resolved",
                         oc="conflict" if i % 3 == 0 else "other_ok", ir="unknown", pred=pa if i % 3 == 0 else other,
                         p=0.8, cos=0.9, ptm=float(r.uniform(0.05, 0.6)), crop_id=nxt(), kt=["KT5"]))
    # reference-lab source: verified admitted, some target rejected
    for i in range(30 * n_scale):
        key = "lu_img%03d" % i
        rows.append(_box(dom, key, 0, LU, pa, 3, "Palmer", "target", tc="verified", ir="admitted", pred=pa, p=0.95,
                         cos=0.95, crop_id=nxt()))
        if i % 5 == 0:
            rows.append(_box(dom, key, 1, LU, mg, 4, "Morning glory", "target", tc="unknown", ir="unknown",
                             pred=mg, p=0.3, cos=0.9, fail="p_below_tau", crop_id=nxt()))
    # anonymous no-name source: conflicts (G1 noinfo) and other_ok (G4 noinfo); one class of 120 boxes
    for i in range(120 * n_scale):
        key = "an_img%03d" % (i // 2)
        conflict = i % 4 == 0
        rows.append(_box(dom, key, i % 2, AN, other, "*", "", "no_name", oc="conflict" if conflict else "other_ok",
                         ir="conflict" if conflict else "admitted", pred=mg if conflict else other, p=0.7,
                         cos=0.9, ptm=float(r.uniform(0.01, 0.9)), crop_id=nxt()))
    # numeric card-resolved source: ids 12 and 5 (100+ boxes each) and id 3
    for sid_, n in (("12", 110), ("5", 104), ("3", 40)):
        for i in range(n * n_scale):
            key = "mh_%s_img%03d" % (sid_, i)
            rows.append(_box(dom, key, 0, MH, other, sid_, sid_, "numeric", oc="other_ok",
                             pred=other, p=0.6, cos=0.8, ptm=float(r.uniform(0.01, 0.5)), crop_id=nxt(),
                             kt=["KT6"] if sid_ in ("12", "5") else []))
    # small boxes
    for i in range(6):
        rows.append(_box(dom, "nd_img%03d" % i, 9, ND, other, 5, "Redroot pigweed", "taxon_resolved",
                         size="small", oc="n/a", ir="unknown"))
    rows.sort(key=lambda x: x["id"])
    return rows


def known_truth(dom):
    """KT1-KT7 items (runner §4.6) for the synthetic world."""
    out = {}
    targets = dom.target_ids
    other = dom.other["id"]

    def it(uid, kt, crop_id, crop_set, truth, taxon, kind, claimed, source, prov, session=None, role=None, key=None):
        return {"id": uid, "kt": kt, "crop_id": crop_id, "crop_set": crop_set, "truth": truth,
                "truth_taxon": taxon, "truth_kind": kind, "claimed": claimed, "source": source,
                "lab": dom.lab_of(source), "near_dup3": "n:%s" % (key or uid), "provenance": prov,
                "session": session, "role": role}
    out["KT1"] = [it("t1:tc%03d#0" % i, "KT1", 1000 + i, "core", targets[i % 12],
                     dom.target(targets[i % 12])["taxon"], "target", False, REF, "prov:tc%03d" % i,
                     session="sess%d" % (i % 7), key="tc%03d" % i) for i in range(96)]
    out["KT2"] = [it("t2:cp%03d#0" % i, "KT2", 2000 + i, "copy", targets[i % 12],
                     dom.target(targets[i % 12])["taxon"], "target", False, COPY_SRC, "prov:tc%03d" % i,
                     session="sess%d" % (i % 6), key="cp%03d" % i) for i in range(72)]
    out["KT3"] = [it("t2:ts%03d#0" % i, "KT3", 3000 + i, "copy", targets[i % 12],
                     dom.target(targets[i % 12])["taxon"], "target", False,
                     "project_agml__three_season_weed_detection", "prov:tc%03d" % (50 + i % 40),
                     key="ts%03d" % i) for i in range(24)]
    rows = world_rows(dom)
    out["KT4"] = [it(r["id"], "KT4", r["crop_id"], "pool", r["label"], dom.target(r["label"])["taxon"], "target",
                     True, r["source"], r["provenance"], key=r["key"]) for r in rows if "KT4" in r["kt"]]
    out["KT5"] = [it(r["id"], "KT5", r["crop_id"], "pool", other, "Amaranthus retroflexus", "attractor", True,
                     r["source"], r["provenance"], key=r["key"]) for r in rows if "KT5" in r["kt"]]
    out["KT6"] = []
    for r in rows:
        if "KT6" in r["kt"]:
            tid = dom.class_id("Sicklepod") if r["src_id"] == "12" else dom.class_id("MorningGlory")
            out["KT6"].append(it(r["id"], "KT6", r["crop_id"], "pool", tid, dom.target(tid)["taxon"], "target",
                                 True, r["source"], r["provenance"], role="calibration", key=r["key"]))
    kt7 = []
    n = 0
    attractors = dom.attractors
    for t in targets:
        for j in range(12):
            role = "exemplar" if j < 6 else ("g0" if (j - 6) % 3 == 2 else "sentinel")
            kt7.append(it("t7:o%04d/p%d" % (n, j), "KT7", 5000 + n, "kt7", t, dom.target(t)["taxon"], "target",
                          False, "kt7", "prov:kt7|o%04d" % n, role=role))
            n += 1
    for a in attractors:
        for j in range(12):
            role = "exemplar" if j < 6 else ("g0" if (j - 6) % 3 == 2 else "sentinel")
            kt7.append(it("t7:o%04d/p%d" % (n, j), "KT7", 5000 + n, "kt7", None, a["taxon"], "attractor",
                          False, "kt7", "prov:kt7|o%04d" % n, role=role))
            n += 1
    out["KT7"] = kt7
    return out


def build_world(fd, dom, prereg_path):
    """Write the census outputs of the synthetic world into fd; returns the
    FakeAdapter."""
    import csv
    sys.path.insert(0, str(PKG_ROOT))
    from weed_optimizer_framework.tools import funnel as F
    from weed_optimizer_framework.tools.funnel import domain as D, ledger as L
    fd = pathlib.Path(fd)
    pre = D.load_prereg(prereg_path)
    rows = world_rows(dom)
    F.write_jsonl_atomic(fd / "ledger.jsonl", rows)
    names = {}
    for r in rows:
        k = (r["source"], r["src_id"])
        e = names.setdefault(k, {"source": r["source"], "src_id": r["src_id"], "name": r["src_name"],
                                 "key": r["src_name"].lower(), "status_v1": "x", "status_v2": r["name_status_v2"],
                                 "via": "none", "taxon": None, "rank": None, "boxes": 0, "conflicts": 0})
        e["boxes"] += 1
        if r["src_name"] == "Redroot pigweed":
            e["taxon"], e["rank"], e["via"] = "Amaranthus retroflexus", "species", "vernacular"
    ns = {"format": "funnel-name-status/2", "frozen": True, "names": [names[k] for k in sorted(names)],
          "frames": dom.raw["names"]["frames"]}
    ns_sha = F.write_json_atomic(fd / "name_status_v2.json", ns)
    census = {"format": "funnel-census/1", "domain": dom.name, "prereg": pre.record(),
              "reconciliation": {"ok": True, "checks": []},
              "name_status_v2": {"path": str(fd / "name_status_v2.json"), "sha256": ns_sha},
              "h5a": {"twins": 12, "dropped_twins_with_target_box_kept_lacks": {"images": 3, "boxes": 7},
                      "dropped_twins_more_informative_names": {"images": 1}},
              "h12": {"present": 18, "total": 20, "items": []},
              "train_core_boxes_per_class": {n: (50 if n in ("CutleafGroundcherry", "Goosegrass", "Sicklepod") else 400)
                                             for n in dom.target_names},
              "totals": {"small_boxes": sum(1 for r in rows if r["path"]["S6"] == "small")}}
    F.write_json_atomic(fd / "census_v1.json", census)
    ident = {"admit_summary": sha("a"), "pool_summary": sha("b"), "select_summary": sha("c"), "calibration": sha("d")}
    led = L.new(dom, {"admit_summary": sha("a")}, "census", "v2", ident, prereg=pre)
    for st in dom.stages:
        L.add_stage(led, {"id": st["id"], "filter": st.get("filter", st["id"]), "version": "v1", "unit": st["unit"],
                          "role": st["role"], "depends_on": st.get("depends_on", []), "recoverable": st["recoverable"],
                          "guard": st["guard"]})
    L.write(fd / "funnel_ledger.json", led)
    with open(fd / "guard_pairs_v1.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["pair_id", "kind", "source", "rel", "dhash", "split", "eval_key", "eval_image", "bits",
                    "sha_a", "sha_b", "sha_differs"])
        for i in range(13):
            w.writerow(["p:%s|images/t%02d.jpg|ood23|e%02d" % (AN, i, i), "near_eval", AN, "images/t%02d.jpg" % i, 1,
                        "ood23", "e%02d" % i, "", i % 7, "a", "b", 1])
        for i in range(30):
            w.writerow(["p:%s|images/q%02d.jpg|ood23|f%02d" % (PER, i, i), "near_eval", PER, "images/q%02d.jpg" % i, 1,
                        "ood23", "f%02d" % i, "", (i * 2) % 7, "a", "b", 1])
        for i in range(8):
            w.writerow(["p:%s|images/d%02d.jpg|dup|k%02d" % (ND2, i, i), "exact_dup", ND2, "images/d%02d.jpg" % i, 1,
                        "dup", "k%02d" % i, "", 0, "a", "b" if i % 2 else "a", int(i % 2 == 1)])
        for i in range(5):
            w.writerow(["p:%s|images/c%02d.jpg|train_core|tc%03d" % (LU, i, i), "cwd12_copy", LU, "images/c%02d.jpg" % i,
                        1, "train_core", "tc%03d" % i, "", 2, "a", "b", 1])
    with open(fd / "leak_pairs_v1.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["set", "key", "eval_split", "eval_key", "cos", "bits", "variant", "kind"])
        for i in range(25):
            w.writerow([REF, "tc%03d" % i, MH, "mh_12_img%03d" % i, 0.3, 8, "id", "negative"])
    return FakeAdapter(known_truth(dom), dom.other["id"])


def hidden_truth(dom, row):
    """The class name a truthful reference labeller gives a pool box of the
    world: target labels are right; the card-resolved numeric ids are their
    card class; half the anonymous conflicts are the predicted target; every
    other box is another plant."""
    if row is None:
        return "other"
    if dom.is_target(row["label"]):
        return dom.class_name(row["label"])
    if row["source"] == MH and row["src_id"] == "12":
        return "Sicklepod"
    if row["source"] == MH and row["src_id"] == "5":
        return "MorningGlory"
    if row["source"] == AN and row["path"]["S9"] == "conflict" and row["box"] == 0:
        return dom.class_name(row["pred"])
    return "other"


def write_step1(step1, dom):
    """The Step 1 files estimate reads: verifier thresholds, the in-domain
    calibration record and the pool summary's listed images."""
    from weed_optimizer_framework.tools import funnel as F
    step1 = pathlib.Path(step1)
    tau = {n: 0.5 for n in dom.target_names}
    tau[dom.other["name"]] = 0.3
    sig = {n: 0.8 for n in dom.class_names}
    F.write_json_atomic(step1 / "verifier" / "thresholds.json", {"tau_p": tau, "sigma": sig})
    F.write_json_atomic(step1 / "calibration.json", {"cwd12_copies": {"current_join": {"overall": {
        "verdicts": {"verified": 500}, "verified_precision": 0.998}}}})
    F.write_json_atomic(step1 / "pool_summary.json", {"per_slug": {MH: {"images_listed": 250}}})


def fake_dinov2(crop_ids):
    """Deterministic 8-d features with two visible clusters."""
    import numpy as np
    out = []
    for c in crop_ids:
        r = np.random.default_rng(int(c) + 11)
        v = r.normal(size=8)
        v[int(c) % 2] += 6.0
        out.append(v)
    return np.array(out)
