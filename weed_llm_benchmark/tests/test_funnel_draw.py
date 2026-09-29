#!/usr/bin/env python3
"""The seeded stratified draw and the sample lock (contract §5.2, F6; runner
§4.7-4.9, §5.1.5).

Pinned, on the synthetic world:
  * every sample row's seed text regenerates its stratum's draw exactly
    (recompute), G0 rows from the key;
  * pi is n_h/N_h for srs rows, 1 - (1 - u_h)(1 - p_i) for G4,
    1 - (1 - pi_G2v)(1 - pi_G2a) for units in both joint frames, and
    (m/M)(n_u/N_u) for G3;
  * the Horvitz-Thompson sum of 1/pi over a frame's sample has expectation N
    (200 seeded redraws, within 3 standard errors) for the G4 dual design, the
    joint G2v/G2a design and the G3 two-stage design with a unit draw;
  * G0 rows show no source, key, lab or truth; the key holds the planted
    share and each G0 item's part;
  * sentinels are 3 per 12 pool items (a sheet holds 15 items, 3 of them
    sentinels), pair sentinels 3 per 12 G5 items; G5 and pair sentinels are
    sheet_class eval, everything else pool;
  * the lock amendment carries the sha256 of sample_v1.csv, the key, the
    frames and name_status_v2.json, the frame sizes and the confirmatory
    frame sizes (H2a, H2b, H4 named, H4 no-information);
  * a second draw with the same inputs is a no-op; with a changed ledger it
    refuses (SampleLocked), with force too; load_sample refuses a sample that
    changed after the lock;
  * a frame below its pre-registered minimum refuses the draw (DrawError),
    and nothing is written or locked;
  * the sample is sorted by group, stratum and draw rank, one row per item.

Run:  python3 tests/test_funnel_draw.py
"""
import copy
import json
import math
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import funnel_stats_world as W  # noqa: E402

TMP = W.setup("funnel_draw_")
check, raises = W.check, W.raises

import numpy as np  # noqa: E402
from weed_optimizer_framework.tools import funnel as F  # noqa: E402
from weed_optimizer_framework.tools.funnel import domain as D  # noqa: E402
from weed_optimizer_framework.tools.funnel import draw as DR  # noqa: E402
from weed_optimizer_framework.tools.funnel import strata as S  # noqa: E402

fd = TMP / "inc" / "funnel"
PREREG = fd / "prereg_v1.json"
GROUPS = {"G0": [20, 10], "G1": [30, 10], "G2": [30, 10], "G2v": [8, 4], "G2a": [8, 4], "G3": [60, 20],
          "G4": [40, 20], "G5": [20, 10], "sentinels": [400, None], "KT7_sentinels": [150, None]}


def set_groups(path, groups, g4_uniform=20):
    raw = json.loads(path.read_text())
    raw["sampling"]["groups"] = groups
    raw["sampling"]["G4_uniform"] = g4_uniform
    path.write_text(json.dumps(raw, indent=1))


dom = D.load("weed")
print("minimum refusal")
bad = copy.deepcopy(GROUPS)
bad["G2v"] = [8, 500]
set_groups(PREREG, bad)
adapter = W.build_world(fd, dom, PREREG)
check("a frame below its minimum refuses", raises(lambda: DR.draw(PREREG, fd, adapter, dinov2=W.fake_dinov2),
                                                  F.DrawError))
check("... nothing written, nothing locked", not (fd / "sample_v1.csv").exists() and
      D.load_prereg(PREREG).sample_lock is None)
bad = copy.deepcopy(GROUPS)
bad["G3"] = [60, 70]          # the G3 frame holds 214 boxes, but its two units give at most 60 items
set_groups(PREREG, bad)
adapter = W.build_world(fd, dom, PREREG)
err = []
try:
    DR.draw(PREREG, fd, adapter, dinov2=W.fake_dinov2)
except F.DrawError as e:
    err.append(str(e))
check("a design that draws fewer items than the minimum from a large enough frame refuses",
      err and "G3" in err[0] and "design draws 60" in err[0], err)
check("... nothing written, nothing locked", not (fd / "sample_v1.csv").exists() and
      D.load_prereg(PREREG).sample_lock is None)
set_groups(PREREG, GROUPS)
check("a census made under another prereg core refuses the draw (StaleInput, runner §3.3)",
      raises(lambda: DR.draw(PREREG, fd, adapter, dinov2=W.fake_dinov2), F.StaleInput))
adapter = W.build_world(fd, dom, PREREG)
pre0 = D.load_prereg(PREREG)

print("draw and lock")
res = DR.draw(PREREG, fd, adapter, dinov2=W.fake_dinov2, testing=True)
pre = D.load_prereg(PREREG)
lock = pre.sample_lock
check("drawn and locked (the lock takes the next free id after the prereg's amendment A2: A3)",
      res["status"] == "drawn" and lock is not None and lock["id"] == D.next_amendment_id(pre0) == "A3")
check("core sha unchanged by the lock", pre.core_sha256 == pre0.core_sha256)
for name, fn in (("sample", "sample_v1.csv"), ("key", "sample_v1_key.jsonl"), ("frames", "frames_v1.json"),
                 ("name_status_v2", "name_status_v2.json")):
    check("lock carries the %s sha256" % name, lock["%s_sha256" % name] == F.file_record(fd / fn)["sha256"])
check("lock names the prereg core", lock["prereg_core_sha256"] == pre0.core_sha256)
frames_doc = json.loads((fd / "frames_v1.json").read_text())
check("lock frame sizes", lock["frame_sizes"] == {g: v["N"] for g, v in frames_doc["groups"].items()})
g1i, g4i = frames_doc["groups"]["G1"]["info"], frames_doc["groups"]["G4"]["info"]
check("confirmatory frame sizes", lock["confirmatory_frames"] == {
    "H2a": g1i["frame_N"].get("noinfo", 0), "H2b": g1i["frame_N"].get("named", 0),
    "H4_named": g4i["frame_N"].get("named", 0), "H4_noinfo": g4i["frame_N"].get("noinfo", 0)})
check("frames header", frames_doc["format"] == "funnel-frames/1" and frames_doc["prereg"]["core_sha256"] ==
      pre0.core_sha256 and "ledger_jsonl" in frames_doc["inputs"])

sample = DR.load_sample(fd / "sample_v1.csv", pre)
key = DR.load_key(fd / "sample_v1_key.jsonl", pre)
frames = S.load_frames(fd / "frames_v1.json")
check("sample columns", list(sample[0]) == list(DR.SAMPLE_FIELDS))
check("one row per item", len({r["item_id"] for r in sample}) == len(sample))
order = [(r["group"], r["stratum"], int(r["draw_rank"])) for r in sample]
check("sorted by group, stratum, draw rank", order == sorted(order))
check("item ids are opaque hashes", all(len(r["item_id"]) == 16 for r in sample))

print("recompute")
bad_rows = [r for r in sample if not DR.recompute(r, frames, key)["ok"]]
check("every row's seed text regenerates its draw (%d rows)" % len(sample), not bad_rows,
      [(r["group"], r["stratum"], r["seed_text"]) for r in bad_rows[:3]])
r0 = dict([r for r in sample if r["group"] == "G2"][0])
r0["draw_rank"] = str(int(r0["draw_rank"]) + 1)
check("a wrong draw rank does not recompute", not DR.recompute(r0, frames, key)["ok"])
r0 = dict([r for r in sample if r["group"] == "G2"][0])
r0["seed_text"] = "funnel/v1/G2/elsewhere"
check("a foreign seed text does not recompute", not DR.recompute(r0, frames, key)["ok"])

print("inclusion probabilities")
docg = frames_doc["groups"]
ok = True
for r in sample:
    g = r["group"]
    if g in ("G1", "G2", "G5", "identity", "sentinel", "pair_sentinel"):
        st = docg[g]["strata"][r["stratum"]]
        ok &= abs(float(r["pi"]) - st["n_planned"] / float(st["N"])) < 1e-12
check("srs rows: pi = n_h / N_h", ok)
g4 = [r for r in sample if r["group"] == "G4"]
ok = all(abs(float(r["pi"]) - (1 - (1 - json.loads(r["pi_parts"])["uniform"]) *
                                (1 - json.loads(r["pi_parts"])["prio"]))) < 1e-12 for r in g4)
check("G4 rows: pi = 1 - (1 - u)(1 - p)", g4 and ok)
units_g4 = {u["unit_id"]: u for u in frames["groups"]["G4"]}
S_ = sum(float(u["score"]) for u in units_g4.values())
ok = all(abs(json.loads(r["pi_parts"])["prio"] - min(1.0, docg["G4"]["info"]["prio"]["n"] *
                                                    float(units_g4[r["unit_id"]]["score"]) / S_)) < 1e-12 for r in g4)
check("G4 prio part = min(1, n s_i / sum s)", ok)
joint = [r for r in sample if r["group"] in ("G2v", "G2a")]
ok = True
for r in joint:
    parts = json.loads(r["pi_parts"])
    ok &= abs(float(r["pi"]) - (1 - (1 - parts["G2v"]) * (1 - parts["G2a"]))) < 1e-12
check("joint rows: pi = 1 - (1 - pi_G2v)(1 - pi_G2a)", joint and ok)
both = [r for r in joint if json.loads(r["pi_parts"])["G2v"] > 0 and json.loads(r["pi_parts"])["G2a"] > 0]
check("units in both frames are listed once", both and len({r["unit_id"] for r in joint}) == len(joint))
g3 = [r for r in sample if r["group"] == "G3"]
info3 = docg["G3"]["info"]
ok = all(abs(float(r["pi"]) - (info3["m"] / float(info3["M"])) * (docg["G3"]["strata"][r["stratum"]]["n_planned"] /
                                                                  float(docg["G3"]["strata"][r["stratum"]]["N"])))
         < 1e-12 for r in g3)
check("G3 rows: pi = (m/M)(n_u/N_u)", g3 and ok)

print("HT expectation over redraws")
fr = S.frames(pre, dom, fd, adapter, dinov2=W.fake_dinov2)


def ht_mean(draw_fn, frame_ids, reps=200):
    tot = []
    for rep in range(reps):
        picks = draw_fn("funnel/test/redraw/%d" % rep)
        tot.append(sum(1.0 / p["pi"] for p in picks if p["unit"]["unit_id"] in frame_ids))
    a = np.array(tot)
    return a.mean(), a.std(ddof=1) / math.sqrt(len(a))


ids4 = {u["unit_id"] for u in fr["groups"]["G4"]["units"]}
m, se = ht_mean(lambda pfx: DR.draw_g4(fr["groups"]["G4"], pfx), ids4)
check("G4 dual: E[sum 1/pi] = N = %d (%.1f +- %.1f)" % (len(ids4), m, se), abs(m - len(ids4)) <= 3 * se + 1e-9)
for g in ("G2v", "G2a"):
    ids = {u["unit_id"] for u in fr["groups"][g]["units"]}
    m, se = ht_mean(lambda pfx: DR.draw_joint(fr["groups"], ("G2v", "G2a"), pfx), ids)
    check("joint %s: E[sum 1/pi] = N = %d (%.2f +- %.2f)" % (g, len(ids), m, se),
          abs(m - len(ids)) <= 3 * se + 1e-9)
grp3 = copy.deepcopy(fr["groups"]["G3"])
S._plan_g3(grp3, 20)
ids3 = {u["unit_id"] for u in grp3["units"]}
check("G3 test plan draws 1 of 2 units", grp3["info"]["m"] == 1 and not grp3["info"]["take_all_units"])
m, se = ht_mean(lambda pfx: DR.draw_g3(grp3, pfx), ids3)
check("G3 two-stage: E[sum 1/pi] = N = %d (%.1f +- %.1f)" % (len(ids3), m, se), abs(m - len(ids3)) <= 3 * se + 1e-9)

print("G0 and the key")
g0 = [r for r in sample if r["group"] == "G0"]
check("G0 rows show no source, key, lab, pi or truth", g0 and all(
    r["source"] == "" and r["image_key"] == "" and r["lab"] == "" and r["pi"] == "" and r["kt"] == ""
    and r["unit_id"].startswith("G0:") and r["seed_text"] == "funnel/v1/G0" for r in g0))
planted = [k for k in key if "planted" in k]
check("the key holds the planted share", len(planted) == 1 and 0.2 <= planted[0]["planted"]["share_drawn"] <= 0.8
      and planted[0]["planted"]["seed_text"] == "funnel/v1/G0/share")
kmap = {k["item_id"]: k for k in key if "item_id" in k}
check("each G0 key row names its part and real unit", all(kmap[r["item_id"]]["g0"]["part"] in
                                                           ("pos_in_domain", "pos_independent", "neg_independent")
                                                           and not kmap[r["item_id"]]["unit_id"].startswith("G0:")
                                                           for r in g0))
n_pos = sum(1 for r in g0 if kmap[r["item_id"]]["g0"]["part"].startswith("pos"))
check("the realised share is the key's share", abs(planted[0]["planted"]["share"] - n_pos / float(len(g0))) < 1e-12)
check("truth only for planted, sentinel, identity and pair-sentinel items",
      all(kmap[r["item_id"]]["truth"] is None for r in sample if r["group"] not in DR.TRUTH_GROUPS))

print("sentinels and sheet classes")
n_pool = sum(1 for r in sample if r["group"] in S.POOL_GROUPS)
n_g5 = sum(1 for r in sample if r["group"] == "G5")
n_sent = sum(1 for r in sample if r["group"] == "sentinel")
n_pair = sum(1 for r in sample if r["group"] == "pair_sentinel")
check("sentinels: 3 per 12 pool items (%d for %d)" % (n_sent, n_pool), n_sent == 3 * math.ceil(n_pool / 12.0))
check("pair sentinels: 3 per 12 G5 items (%d for %d)" % (n_pair, n_g5), n_pair == 3 * math.ceil(n_g5 / 12.0))
check("eval sheet class for G5 and pair sentinels only",
      all((r["sheet_class"] == "eval") == (r["group"] in ("G5", "pair_sentinel")) for r in sample))
sent_kts = {r["kt"] for r in sample if r["group"] == "sentinel"}
check("sentinels come from the weighted known-truth sets", sent_kts <= set(dom.raw["sampling"]["sentinel_weights"]))
check("no independent-set exemplar or g0 photo is a sentinel",
      not any(r["unit_id"].startswith("t7:") and [it for it in W.known_truth(dom)["KT7"]
                                                    if it["id"] == r["unit_id"]][0]["role"] != "sentinel"
              for r in sample if r["group"] == "sentinel"))

known0 = W.known_truth(dom)
ex_by_cls = S.exemplar_sessions_by_class(dom, S.exemplar_sessions(dom, known0))
kt1 = {it["id"]: it for it in known0["KT1"]}
kt1_sent = [kt1[r["unit_id"]] for r in sample if r["group"] == "sentinel" and r["unit_id"] in kt1]
check("KT1 sentinels exclude their class's board-exemplar sessions (%d KT1 sentinels)" % len(kt1_sent),
      kt1_sent and not any(it["session"] in ex_by_cls.get(int(it["truth"]), ()) for it in kt1_sent))

print("reruns and the lock")
before = {n: F.file_record(fd / n)["sha256"] for n in ("sample_v1.csv", "sample_v1_key.jsonl", "frames_v1.json")}
again = DR.draw(PREREG, fd, adapter, dinov2=W.fake_dinov2, testing=True)
check("a second draw with the same inputs is a no-op", again["status"] == "no-op" and
      {n: F.file_record(fd / n)["sha256"] for n in before} == before
      and len(D.load_prereg(PREREG).amendments) == len(pre0.amendments) + 1)
changed_known = W.known_truth(dom)
changed_known["KT7"] = changed_known["KT7"][1:]
check("a locked rerun whose known truth changed refuses (SampleLocked)",
      raises(lambda: DR.draw(PREREG, fd, W.FakeAdapter(changed_known), dinov2=W.fake_dinov2), F.SampleLocked))
led = fd / "ledger.jsonl"
keep = led.read_bytes()
led.write_bytes(keep + b"\n")
check("a changed ledger refuses (SampleLocked)", raises(lambda: DR.draw(PREREG, fd, adapter, dinov2=W.fake_dinov2),
                                                        F.SampleLocked))
check("force does not lift the lock", raises(lambda: DR.draw(PREREG, fd, adapter, force=True,
                                                              dinov2=W.fake_dinov2), F.SampleLocked))
led.write_bytes(keep)
txt = (fd / "sample_v1.csv").read_text()
(fd / "sample_v1.csv").write_text(txt + "\n")
check("load_sample refuses a changed sample", raises(lambda: DR.load_sample(fd / "sample_v1.csv", pre), F.StaleInput))
check("draw refuses when the locked files changed", raises(lambda: DR.draw(PREREG, fd, adapter,
                                                                           dinov2=W.fake_dinov2), F.SampleLocked))
(fd / "sample_v1.csv").write_text(txt)
check("load_sample without a lock refuses", raises(lambda: DR.load_sample(fd / "sample_v1.csv", pre0), F.DrawError))
locked_bytes = PREREG.read_bytes()
edited = json.loads(locked_bytes)
edited["hypotheses"]["H1"]["supported"] = "LB>=0.50 overall and for Ragweed"
PREREG.write_text(json.dumps(edited, indent=1))
pre_edited = D.load_prereg(PREREG)
check("a prereg edited outside its amendments after the lock: draw refuses (SampleLocked)",
      raises(lambda: DR.draw(PREREG, fd, adapter, dinov2=W.fake_dinov2), F.SampleLocked))
check("... load_sample and load_key refuse (StaleInput)",
      raises(lambda: DR.load_sample(fd / "sample_v1.csv", pre_edited), F.StaleInput) and
      raises(lambda: DR.load_key(fd / "sample_v1_key.jsonl", pre_edited), F.StaleInput))
PREREG.write_bytes(locked_bytes)
check("the restored prereg loads the sample again", len(DR.load_sample(fd / "sample_v1.csv", D.load_prereg(PREREG)))
      == len(sample))

print("unlocked reruns")
tmp2 = TMP / "second"
(tmp2).mkdir()
p2 = tmp2 / "prereg_v1.json"
p2.write_text(json.dumps(pre0.raw, indent=1))
for n in ("census_v1.json", "ledger.jsonl", "name_status_v2.json", "funnel_ledger.json", "guard_pairs_v1.csv",
          "leak_pairs_v1.csv", "leak_pairs_v2.csv", "leak_v2.json"):
    shutil.copy(fd / n, tmp2 / n)
census = json.loads((tmp2 / "census_v1.json").read_text())
census["name_status_v2"]["path"] = str(tmp2 / "name_status_v2.json")
F.write_json_atomic(tmp2 / "census_v1.json", census)
res2 = DR.draw(p2, tmp2, adapter, dinov2=W.fake_dinov2, testing=True)
check("the same world draws the same sample elsewhere", F.file_record(tmp2 / "sample_v1.csv")["sha256"] ==
      before["sample_v1.csv"] and F.file_record(tmp2 / "sample_v1_key.jsonl")["sha256"] == before["sample_v1_key.jsonl"])

W.finish()
