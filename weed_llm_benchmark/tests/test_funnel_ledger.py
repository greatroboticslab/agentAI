#!/usr/bin/env python3
"""The funnel ledger funnel-ledger/1 and the unit rows of ledger.jsonl
(contract §8.2; runner §4.2-4.3, §5.1.3).

Pinned:
  * a hand-built ledger over the domain's stage table validates, writes and
    loads back unchanged;
  * each rule refuses a broken copy: in != kept + sum(discarded), a
    kept_by_label that does not sum to kept, a guard stage marked
    recoverable, a depends_on naming a later or unknown stage, an unknown
    role or unit, a repeated id, a fingerprint that does not match its
    identity inputs, label-space kinds that do not sum to the boxes;
  * add_stage refuses the same errors before they enter a ledger;
  * fingerprint is the sha256 of the canonical identity inputs (order free);
  * unaudited_dependencies on the stage table returns (S12, S8) among its
    pairs and drops a pair once its dependency carries an audit;
  * attach_audit records the audit file's sha256 and FN rate per stage and
    refuses an invalid audit or one made on another ledger;
  * validate_unit_row accepts a box row and a pre-pool image row and
    refuses a wrong first/sole cause, failed stages out of stage order, a
    path value outside the adapter's vocabulary and a malformed id;
  * iter_units streams a 100,000-row file with a tracemalloc peak under 50 MB.

Run:  python3 tests/test_funnel_ledger.py
"""
import copy
import json
import pathlib
import sys
import tracemalloc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import funnel_stats_world as W  # noqa: E402

TMP = W.setup("funnel_ledger_")
check, raises = W.check, W.raises

from weed_optimizer_framework.tools import funnel as F  # noqa: E402
from weed_optimizer_framework.tools.funnel import domain as D  # noqa: E402
from weed_optimizer_framework.tools.funnel import ledger as L  # noqa: E402

dom = D.load("weed")
pre = D.load_prereg(TMP / "inc" / "funnel" / "prereg_v1.json")
IDENT = {"admit_summary": W.sha("a"), "pool_summary": W.sha("p"), "select_summary": W.sha("s"),
         "calibration": W.sha("c")}


def build():
    led = L.new(dom, {"admit_summary": W.sha("a")}, "census", "v2", IDENT, prereg=pre, testing=True)
    for st in dom.stages:
        rec = {"id": st["id"], "filter": st.get("filter", st["id"]), "version": "v1", "unit": st["unit"],
               "role": st["role"], "depends_on": st.get("depends_on", []), "recoverable": st["recoverable"],
               "guard": st["guard"]}
        if st["role"] == "target_check":
            rec.update({"in": 100, "kept": 30, "discarded": {"conflict": 40, "unknown": 30},
                        "kept_by_label": {"Waterhemp": 20, "Ragweed": 10},
                        "discarded_by_label": {"Ragweed": {"conflict": 40, "unknown": 30}},
                        "calibration": {"known_truth_sets": [{"id": "KT2", "sha256": W.sha("k"), "sources": ["x"],
                                                              "domain_score": [0.1, 0.9], "domain_basis": "median"}],
                                        "precision": 1.0, "recall": 0.95, "domains_covered": ["reference"]}})
        if st["role"] == "other_check":
            rec.update({"in": 500, "kept": 450, "discarded": {"conflict": 50},
                        "by_label_pred": {"OtherPlant|PalmerAmaranth": 50}})
        L.add_stage(led, rec)
    led["label_spaces"] = {"src_a": {"classes": 3, "kinds": {"named": 5, "numeric": 3, "none": 2, "generic": 0},
                                     "boxes": 10}}
    led["domain_scores"] = {"src_a": 0.05, "reference": {"q05": 0.09, "q50": 0.4}}
    led["reject_class"] = {"stage": "S8b", "eligible_top_cell": "src_a|x", "drawn_top_cell": None,
                           "sample_sources": None}
    return led


print("build, write, load")
led = build()
check("hand-built ledger validates", L.validate(led) == [], L.validate(led)[:3])
p = TMP / "funnel_ledger.json"
sha = L.write(p, led)
back = L.load(p)
check("write/load round trip", back == json.loads(F.json_text(led)) and sha == F.file_record(p)["sha256"])
check("header keys present", all(k in led for k in ("built_utc", "prereg", "contract", "domain_config")))
check("inputs are {name: sha256}", led["inputs"] == {"admit_summary": W.sha("a")})
check("stage_by_role", [s["id"] for s in L.stage_by_role(led, "target_check")] == ["S8"])
check("stage_by_role refuses an unknown role", raises(lambda: L.stage_by_role(led, "magic"), F.LedgerError))
check("recoverable_stages = recoverable true", L.recoverable_stages(led) == dom.recoverable_stages())

print("fingerprint")
check("fingerprint = sha256 of canonical identity inputs",
      L.fingerprint(IDENT) == F.sha256_bytes(F.canonical_json(IDENT).encode("utf-8")))
check("fingerprint is order free", L.fingerprint(dict(reversed(list(IDENT.items())))) == L.fingerprint(IDENT))
check("fingerprint refuses a non-sha", raises(lambda: L.fingerprint({"a": "nope"}), F.LedgerError))
check("ledger fingerprint", led["fingerprint"] == L.fingerprint(IDENT))

print("validate refusals")


def stage(led_, sid):
    return [s for s in led_["stages"] if s["id"] == sid][0]


def broken(edit):
    b = copy.deepcopy(led)
    edit(b)
    return L.validate(b)


cases = [
    ("in != kept + discarded", lambda b: stage(b, "S8").update({"in": 101}), "in 101"),
    ("kept_by_label sum", lambda b: stage(b, "S8")["kept_by_label"].update({"Ragweed": 11}), "kept_by_label"),
    ("guard recoverable", lambda b: stage(b, "S4").update({"recoverable": True}), "guard"),
    ("depends_on a later stage", lambda b: stage(b, "S2").update({"depends_on": ["S10"]}), "earlier"),
    ("depends_on an unknown stage", lambda b: stage(b, "S2").update({"depends_on": ["S99"]}), "earlier"),
    ("unknown role", lambda b: stage(b, "S2").update({"role": "magic"}), "role"),
    ("unknown unit", lambda b: stage(b, "S2").update({"unit": "pixel"}), "unit"),
    ("repeated id", lambda b: b["stages"].append(copy.deepcopy(stage(b, "S2"))), "duplicate"),
    ("fingerprint mismatch", lambda b: b.update({"fingerprint": W.sha("other")}), "fingerprint"),
    ("label-space kinds sum", lambda b: b["label_spaces"]["src_a"]["kinds"].update({"none": 3}), "kinds sum"),
    ("unknown derivation", lambda b: b.update({"derivation": "guess"}), "derivation"),
    ("missing target classes", lambda b: b.update({"target_classes": []}), "target_classes"),
    ("reject_class names no stage", lambda b: b["reject_class"].update({"stage": "S99"}), "reject_class"),
]
for name, fn, word in cases:
    probs = broken(fn)
    check("refuses %s" % name, probs and any(word in x for x in probs), probs[:3])
bad = copy.deepcopy(led)
stage(bad, "S8")["in"] = 7
check("write refuses an invalid ledger", raises(lambda: L.write(TMP / "bad.json", bad), F.LedgerError))
(TMP / "bad.json").write_text(json.dumps(bad))
check("load refuses an invalid ledger", raises(lambda: L.load(TMP / "bad.json"), F.LedgerError))

print("add_stage")
fresh = L.new(dom, {}, "summaries", "v1", IDENT)
L.add_stage(fresh, {"id": "A", "filter": "f", "version": "v", "unit": "box", "role": "read", "depends_on": [],
                    "recoverable": False, "guard": False})
check("optional keys default", stage(fresh, "A")["audit"] is None and stage(fresh, "A")["discarded"] == {})
for name, st in (("unknown key", {"id": "B", "filter": "f", "version": "v", "unit": "box", "role": "read",
                                  "depends_on": [], "recoverable": False, "guard": False, "colour": 1}),
                 ("guard recoverable", {"id": "B", "filter": "f", "version": "v", "unit": "box", "role": "guard",
                                        "depends_on": [], "recoverable": "sometimes", "guard": True}),
                 ("unknown dependency", {"id": "B", "filter": "f", "version": "v", "unit": "box", "role": "read",
                                         "depends_on": ["Z"], "recoverable": False, "guard": False}),
                 ("repeated id", {"id": "A", "filter": "f", "version": "v", "unit": "box", "role": "read",
                                  "depends_on": [], "recoverable": False, "guard": False}),
                 ("missing role", {"id": "B", "filter": "f", "version": "v", "unit": "box", "depends_on": [],
                                   "recoverable": False, "guard": False})):
    check("add_stage refuses %s" % name, raises(lambda st=st: L.add_stage(fresh, st), F.LedgerError))
check("new refuses an unknown derivation", raises(lambda: L.new(dom, {}, "guess", "v1", IDENT), F.LedgerError))

print("unaudited dependencies")
pairs = L.unaudited_dependencies(led)
check("(S12, S8) among the pairs", ("S12", "S8") in pairs, pairs)
check("only recoverable dependencies", all(stage(led, d)["recoverable"] is True for _s, d in pairs))
led2 = copy.deepcopy(led)
stage(led2, "S8")["audit"] = {"sha256": W.sha("x"), "fn_rate": {"estimate": 0.1, "interval": [0, 1], "n": 3}}
check("an audited dependency drops out", ("S12", "S8") not in L.unaudited_dependencies(led2))

print("attach_audit")
audit = {"format": "funnel-audit/1", "valid": True, "ledger_fingerprint": led["fingerprint"],
         "stages": {"S8": {"fn_rate": {"estimate": 0.4, "interval": [0.3, 0.5], "n": 60, "method": "kg"}}}}
ap = TMP / "audit_v1.json"
F.write_json_atomic(ap, audit)
att = L.attach_audit(led, ap)
check("audit sha and FN rate on the stage", stage(att, "S8")["audit"] == {
    "sha256": F.file_record(ap)["sha256"], "fn_rate": {"estimate": 0.4, "interval": [0.3, 0.5], "n": 60}})
check("other stages untouched", stage(att, "S9")["audit"] is None and stage(led, "S8")["audit"] is None)
check("attached ledger validates", L.validate(att) == [])
F.write_json_atomic(ap, dict(audit, valid=False, calibration_overlap=[{"judge": "J", "stratum": "s"}]))
check("an invalid audit is refused", raises(lambda: L.attach_audit(led, ap), F.LedgerError))
F.write_json_atomic(ap, dict(audit, ledger_fingerprint=W.sha("other")))
check("an audit of another ledger is refused", raises(lambda: L.attach_audit(led, ap), F.LedgerError))

print("unit rows")
ORDER = [s["id"] for s in dom.stages]
VOCAB = {"S2": None, "S3": ("kept", "exact_dup"), "S4": ("pass", "near_eval"), "S5": ("pass", "cwd12_copy"),
         "S6": ("embedded", "small"), "S7": ("target", "other"), "S8": ("verified", "conflict", "unknown", "n/a"),
         "S9": ("other_ok", "conflict", "n/a"), "S10": ("admitted", "conflict", "unknown"), "S11": None,
         "S12": ("evidenced", "not_evidenced", "n/a")}
box = W._box(dom, "k1", 0, W.ND, 5, 8, "Ragweed", "target", tc="unknown", ir="unknown", pred=12, p=0.4, cos=0.5,
             fail="p_below_tau", crop_id=3)
check("a box row validates", L.validate_unit_row(box, ORDER, VOCAB) == [], L.validate_unit_row(box, ORDER, VOCAB))
check("two failed stages have no sole cause", box["failed_stages"] == ["S8", "S10"] and box["sole_cause"] is None)
img = {"id": "d:%s|images/x.jpg" % W.ND, "unit": "image", "source": W.ND, "rel": "images/x.jpg", "dhash": 123,
       "path": {"S2": "pass", "S3": "exact_dup", "S4": "pass", "S5": "pass"}, "near": None, "twin_of": "k1",
       "failed_stages": ["S3"], "first_cause": "S3", "sole_cause": "S3"}
check("an image row validates", L.validate_unit_row(img, ORDER, VOCAB) == [])
for name, edit, word in (
        ("wrong first cause", lambda r: r.update({"first_cause": "S10"}), "first_cause"),
        ("sole cause with two failures", lambda r: r.update({"sole_cause": "S8"}), "sole_cause"),
        ("failed stages out of order", lambda r: r.update({"failed_stages": ["S10", "S8"], "first_cause": "S10"}),
         "order"),
        ("path value outside the vocabulary", lambda r: r["path"].update({"S8": "maybe"}), "not in"),
        ("malformed id", lambda r: r.update({"id": "b:other#0"}), "id"),
        ("unknown key", lambda r: r.update({"colour": 1}), "unknown key"),
        ("non-integer label", lambda r: r.update({"label": "Ragweed"}), "label")):
    r = copy.deepcopy(box)
    edit(r)
    probs = L.validate_unit_row(r, ORDER, VOCAB)
    check("unit row refuses %s" % name, probs and any(word in x for x in probs), probs[:3])
r = copy.deepcopy(img)
r["id"] = "d:x|y"
check("image row refuses a malformed id", any("id" in x for x in L.validate_unit_row(r, ORDER, VOCAB)))

print("iter_units streams")
big = TMP / "ledger.jsonl"
with open(big, "w") as fh:
    for i in range(100000):
        row = copy.deepcopy(box)
        row["id"] = "b:k%06d#0" % i
        row["key"] = "k%06d" % i
        fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    fh.write(json.dumps(img) + "\n")
tracemalloc.start()
n = 0
for row in L.iter_units(big, unit="box"):
    n += 1
_cur, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
check("every box row streamed", n == 100000, n)
check("peak memory under 50 MB (%.1f MB)" % (peak / 1e6), peak < 50e6)
check("unit filter", sum(1 for _ in L.iter_units(big, unit="image")) == 1)
with open(big, "a") as fh:
    fh.write("{broken\n")
check("a malformed line refuses", raises(lambda: sum(1 for _ in L.iter_units(big)), F.LedgerError))

W.finish()
