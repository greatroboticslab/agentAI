#!/usr/bin/env python3
"""funnel/sheets.py (runner docs/FUNNEL_AUDIT_RUNNER.md §4.10, §5.4.1; contract
docs/FUNNEL_AUDIT.md §4.3, DEC-2).

On the synthetic locked sample of tests/funnel_labels_fixtures.sheet_world
(the weed config's two boards, pool items of an unaffiliated source, of the
authoritative lab group and of the reference lab group, independent photos,
reference crops, a copy, named non-target boxes, guard pairs and pair
sentinels):
  * every pool sheet has 15 items with 3 sentinels, at most 4 low-prior (G4)
    items and an expected prevalence of at least 0.2; the pair sheet has 15
    items with 3 pair sentinels;
  * no sheet mixes boards: every item on a sheet shares nothing with its
    board's material;
  * items of the reference lab group, reference crops and the copy land on
    B2 (independent exemplars only); the others on B1;
  * the crop tile is byte-identical to verify._cut_task's crop, and the
    context tile is a letterbox with the box in red;
  * blinding: no sheet JSON holds a source slug, stratum, unit id, class
    name or taxon of the key; the item and sheet keys are the pinned ones;
  * guard pairs and pair sentinels appear only in sheets_v1_cluster/, whose
    index says contains_eval_pixels; sheets_v1/ says it does not;
  * the key's sha256 is in both indexes, and a rerun is a no-op;
  * rendering is deterministic: a second world built the same way renders
    the same image sha256 values;
  * person_items puts a wrong proposal, drawn from the configured attractor
    pairs, on a quarter of the sentinels, asks a fifth of all items blind,
    and person_view hides the anchor flag;
  * a sheet item that is a board exemplar is refused; an item that shares
    material with every board is refused (DisjointnessError).

Needs numpy and PIL (skips otherwise). No network, no GPU.

Run:  python3 tests/test_funnel_sheets.py
"""
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import funnel_labels_fixtures as FX  # noqa: E402

TMP = FX.setup("funnel_sheets_")
FAILURES, SKIPS = [], []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def raises(fn, exc):
    try:
        fn()
    except exc as e:
        return str(e) or type(e).__name__
    except Exception as e:  # noqa: BLE001
        return None if exc is Exception else "WRONG %s: %s" % (type(e).__name__, e)
    return None


def main():
    missing = FX.have("numpy", "PIL")
    if missing:
        print("SKIP: %s not installed" % ", ".join(missing))
        SKIPS.append("sheets")
        return
    import numpy as np
    from PIL import Image
    from weed_optimizer_framework.tools.inc import verify as V
    from weed_optimizer_framework.tools.funnel import domain as D
    from weed_optimizer_framework.tools.funnel import qualify as Q
    from weed_optimizer_framework.tools.funnel import sheets as SH
    from weed_optimizer_framework.tools.funnel import DisjointnessError, SheetError, read_json, read_jsonl
    fd = pathlib.Path(os.environ["INC_DIR"]) / "funnel"
    domain = D.load("weed")
    world = FX.sheet_world(TMP, domain)
    prereg = D.load_prereg(fd / "prereg_v1.json")
    index = SH.run(prereg, domain, fd, world["adapter"])
    pool_dir, eval_dir, key_dir = fd / SH.SHEETS_DIR, fd / SH.CLUSTER_DIR, fd / SH.KEY_DIR
    key = read_jsonl(key_dir / "key.jsonl")
    packing = read_json(key_dir / "packing.json")
    idx_pool = read_json(pool_dir / "index.json")
    idx_eval = read_json(eval_dir / "index.json")
    sheets = {e["sheet_id"]: read_json(pool_dir / ("%s.json" % e["sheet_id"])) for e in idx_pool["sheets"]}
    pair_sheets = {e["sheet_id"]: read_json(eval_dir / ("%s.json" % e["sheet_id"])) for e in idx_eval["sheets"]}
    by_sheet = {}
    for r in key:
        by_sheet.setdefault(r["sheet_id"], []).append(r)
    rl = domain.raw["reference_labeller"]

    # composition
    check("five pool sheets and one pair sheet", len(sheets) == 5 and len(pair_sheets) == 1,
          (len(sheets), len(pair_sheets)))
    for sid, rows in sorted(by_sheet.items()):
        n_sent = sum(1 for r in rows if r["is_sentinel"])
        n_low = sum(1 for r in rows if r["group"] == "G4")
        check("%s: 15 items, 3 sentinels, <= 4 low-prior" % sid,
              len(rows) == 15 and n_sent == 3 and n_low <= rl["sheet"]["max_low_prior"], (len(rows), n_sent, n_low))
    for p in packing["sheets"]:
        if p["kind"] == "mc":
            check("%s: expected prevalence >= 0.2 and 1-3 target sentinels" % p["sheet_id"],
                  p["expected_prevalence"] >= 0.2 and 1 <= p["target_sentinels"] <= 3 and not p["below_min_prevalence"],
                  p)
    check("every sampled item is on exactly one sheet",
          sorted(r["item_id"] for r in key) == sorted(r["item_id"] for r in world["sample"]))

    # boards
    plans = SH.board_plans(domain, world["known"])
    for sid, rows in sorted(by_sheet.items()):
        boards = {r["board_id"] for r in rows}
        check("%s: one board" % sid, len(boards) == 1, boards)
        b = boards.pop()
        if b is None:
            continue
        bad = []
        for r in rows:
            kts = [x for x in (r["kt"] or "").split(";") if x]
            ind = any(k in domain.independent_sets() for k in kts)
            k = world["keys"].get(r["unit_id"])
            if Q.shares(Q.item_keys(dict(k, id=r["unit_id"]), independent=ind), plans[b]["material"]) != \
                    {kk: [] for kk in Q.KEYS}:
                bad.append(r["unit_id"])
        check("%s: nothing on it shares material with board %s" % (sid, b), not bad, bad[:3])
    board_of = {r["unit_id"]: r["board_id"] for r in key}
    lu_units = ["b:%s#0" % p["key"] for p in world["pool_specs"] if p["source"] == FX.LU]
    check("items of the reference lab group land on B2", all(board_of[u] == "B2" for u in lu_units),
          [board_of[u] for u in lu_units])
    check("reference crops and the copy land on B2",
          all(board_of[it["id"]] == "B2" for it, kt in world["sentinels"] if kt in ("KT1", "KT2"))
          and all(board_of[it["id"]] == "B2" for it in world["identity"]))
    nd_units = ["b:%s#0" % p["key"] for p in world["pool_specs"] if p["source"] == FX.ND and p["group"]]
    check("authoritative-lab items land on B1", all(board_of[u] == "B1" for u in nd_units))
    exemplars = {it["id"] for p in plans.values() for xs in p["exemplars"].values() for it in xs}
    check("B1's target exemplars come from two reference sessions per class",
          all(len(v) == 2 for v in plans["B1"]["sessions"].values()) and plans["B1"]["sessions"])
    check("no item on a sheet is a board exemplar", not (exemplars & {r["unit_id"] for r in key}))
    bj = read_json(pool_dir / "board_B1.json")
    check("board json: options numbered 1..n as domain.options()",
          [o["n"] for o in bj["options"]] == list(range(1, len(domain.options()) + 1))
          and bj["image"]["sha256"] == FX_sha(pool_dir / bj["image"]["file"]))

    # tiles
    crow = world["crops"][0]
    tile = SH.crop_tile(crow["image"], {"cx": crow["cx"], "cy": crow["cy"], "w": crow["w"], "h": crow["h"]})
    _img, _ids, arrs, err = V._cut_task((crow["image"], [{"crop_id": 0, "cx": crow["cx"], "cy": crow["cy"],
                                                          "w": crow["w"], "h": crow["h"], "W": 0, "H": 0}]))
    check("the crop tile is verify._cut_task's crop, byte for byte",
          err is None and np.array_equal(np.asarray(tile), arrs[0]))
    im = Image.new("RGB", (100, 50), (0, 0, 255))
    lb = np.asarray(SH.letterbox(im, 224, {"cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5}))
    check("letterbox: grey padding, red box edge, blue inside",
          tuple(lb[2, 112]) == SH.GREY and tuple(lb[112, 56]) == SH.RED and tuple(lb[112, 112]) == (0, 0, 255),
          (tuple(lb[2, 112]), tuple(lb[112, 56]), tuple(lb[112, 112])))
    one = sheets[sorted(sheets)[0]]
    it0 = one["items"][0]
    check("tiles sit in the pinned panel layout",
          it0["tiles"]["crop"][2:] == [224, 224] and it0["panel"][2:] == [448, 244]
          and it0["tiles"]["context"][0] == it0["panel"][0] + 224)

    # blinding
    forbidden = set(domain.class_names)
    for r in world["sample"]:
        forbidden.update([r["unit_id"], r["stratum"], r["source"]])
    for r in key:
        forbidden.update([r["unit_id"], r["truth_taxon"]])
    probs = []
    for sid, doc in list(sheets.items()) + list(pair_sheets.items()):
        probs += SH.blind_problems(doc, forbidden)
        text = (pool_dir if sid in sheets else eval_dir).joinpath("%s.json" % sid).read_text()
        body = json.loads(text)
        body.pop("options")
        body.pop("question")
        low = json.dumps(body).lower()
        probs += [f for f in forbidden if f and len(str(f)) >= 3 and str(f).lower() in low]
    check("blinding: no source, stratum, unit id, class name or taxon in any sheet json", not probs, probs[:5])
    leaky = dict(one, source=FX.ND)
    check("blind_problems flags a planted source key", SH.blind_problems(leaky, forbidden))

    # separation
    g5 = {r["item_id"] for r in world["sample"] if r["sheet_class"] == "eval"}
    in_pool = {i["item_id"] for d in sheets.values() for i in d["items"]}
    in_eval = {i["item_id"] for d in pair_sheets.values() for i in d["items"]}
    check("guard pairs and pair sentinels only in sheets_v1_cluster", not (g5 & in_pool) and g5 == in_eval)
    check("indexes: contains_eval_pixels false / true",
          idx_pool["contains_eval_pixels"] is False and idx_eval["contains_eval_pixels"] is True)
    check("the pair sheet has no board and pair options",
          all(d["board"] is None and len(d["options"]) == len(domain.pair_options()) for d in pair_sheets.values()))
    plan = {"sheet_id": "sheet_9999", "placed": []}
    check("a pair sheet is never written to sheets_v1",
          raises(lambda: SH.render_pair_sheet(plan, pool_dir), SheetError))
    ksha = FX_sha(key_dir / "key.jsonl")
    check("the key's sha256 is in both indexes", idx_pool["key_sha256"] == ksha == idx_eval["key_sha256"])
    again = SH.run(prereg, domain, fd, world["adapter"])
    check("a rerun on the same inputs is a no-op", again == index and FX_sha(key_dir / "key.jsonl") == ksha)

    # determinism: a second world, rendered the same way
    shas_1 = sorted((e["sheet_id"], e["image_sha256"], e["json_sha256"]) for e in idx_pool["sheets"] + idx_eval["sheets"])
    tmp2 = FX.pathlib.Path(str(TMP) + "_b")
    shutil.rmtree(tmp2, ignore_errors=True)
    old_env = (os.environ["INC_DIR"], os.environ["REPO"])
    try:
        shutil.copytree(TMP, tmp2)
        fd2 = tmp2 / "inc" / "funnel"
        for d in (SH.SHEETS_DIR, SH.CLUSTER_DIR, SH.KEY_DIR):
            shutil.rmtree(fd2 / d, ignore_errors=True)
        idx2 = SH.run(D.load_prereg(fd2 / "prereg_v1.json"), domain, fd2, world["adapter"])
        e2 = read_json(fd2 / SH.CLUSTER_DIR / "index.json")
        shas_2 = sorted((e["sheet_id"], e["image_sha256"], e["json_sha256"]) for e in idx2["sheets"] + e2["sheets"])
        check("rendering is deterministic (image and json sha256 over two runs)", shas_1 == shas_2)
    finally:
        os.environ["INC_DIR"], os.environ["REPO"] = old_env
        shutil.rmtree(tmp2, ignore_errors=True)

    # person items (L14)
    items = [{"item_id": r["item_id"], "is_sentinel": r["is_sentinel"], "truth": r["truth"],
              "truth_kind": r["truth_kind"], "truth_taxon": r["truth_taxon"]}
             for r in key if r["board_id"] is not None]
    props = {r["item_id"]: 1 for r in key if not r["is_sentinel"] and r["board_id"] is not None}
    rows = SH.person_items(items, props, domain=domain)
    n_sent = sum(1 for i in items if i["is_sentinel"])
    anchors = [r for r in rows if r["anchor"]]
    blind = [r for r in rows if r["mode"] == "blind"]
    check("a quarter of the sentinels carry a wrong proposal", len(anchors) == round(0.25 * n_sent),
          (len(anchors), n_sent))
    check("a fifth of all items are blind multiple choice", len(blind) >= round(0.2 * len(items)),
          (len(blind), len(items)))
    opts = {o["n"]: o for o in domain.options()}
    truth_of = {i["item_id"]: i for i in items}
    ok_pairs = True
    for r in anchors:
        t = truth_of[r["item_id"]]
        o = opts[r["proposal"]]
        if t["truth_kind"] == "target":
            tgt = domain.target(int(t["truth"]))
            allowed = set(tgt.get("siblings") or []) | {a["taxon"] for a in domain.attractors
                                                        if tgt["name"] in (a.get("confused_with") or [])}
            ok_pairs &= (o["answer"] in allowed) or (o["taxon"] in allowed)
        else:
            a = next(a for a in domain.attractors if a["taxon"] == t["truth_taxon"])
            ok_pairs &= o["answer"] in (a.get("confused_with") or [])
    check("wrong proposals come from the configured attractor pairs", ok_pairs and anchors)
    view = SH.person_view(rows)
    check("person_view hides the anchor and the proposal source",
          all(set(v) == {"item_id", "mode", "proposal", "proposal_text"} for v in view))
    check("person_items is deterministic", SH.person_items(items, props, domain=domain) == rows)

    # composition rules on synthetic queues (compose directly)
    def q(group, n, prior, prefix):
        return [{"item_id": "%s%02d" % (prefix, i), "group": group, "stratum": "%s/s" % group, "is_sentinel": False,
                 "truth_kind": None} for i in range(n)], {"%s/s" % group: prior}
    g4, pr4 = q("G4", 20, 0.0, "a")
    sent = [{"item_id": "s%02d" % i, "group": "sentinel", "stratum": "sentinel/x", "is_sentinel": True,
             "truth_kind": "target"} for i in range(30)]
    cs = SH.compose(g4 + sent, {}, pr4, domain, board_id="B1")
    check("compose: at most max_low_prior G4 items on any sheet, even when G4 is all that is left",
          all(s["n_low_prior"] <= rl["sheet"]["max_low_prior"] for s in cs) and len(cs) >= 5,
          [s["n_low_prior"] for s in cs])
    raw2 = json.loads(json.dumps(domain.raw))
    raw2["reference_labeller"]["sheet"]["min_prevalence"] = 0.3
    dom2 = D.Domain(raw2, domain.path, domain.sha256)
    g2, pr2 = q("G2", 30, 0.1, "b")
    g4b, _ = q("G4", 4, 0.0, "c")
    cs2 = SH.compose(g2 + g4b + sent, {}, dict(pr2, **pr4), dom2, board_id="B1")
    check("compose: a sheet short of min_prevalence with every sentinel a target swaps its G4 items for G2 ones",
          cs2[0]["n_low_prior"] == 0 and len(cs2[0]["items"]) == 12, [(s["n_low_prior"], len(s["items"])) for s in cs2])

    # refusals
    kt_all = {k: v for k, v in world["known"].items()}
    g0_row = next(r for r in world["sample"] if r["group"] == "G0")

    class NoKeys(FX.FakeAdapter):
        def unit_keys(self, unit_ids):
            return {}
    no_keys = NoKeys(world["known"], {})
    check("an item whose disjointness keys cannot be found is refused, not boarded as sharing nothing",
          raises(lambda: SH._items([g0_row], world["key"], no_keys, domain, {}), DisjointnessError))
    check("... and with its keys it is accepted",
          len(SH._items([g0_row], world["key"], world["adapter"], domain, {})) == 1)
    ex_one = next(it for xs in plans["B1"]["exemplars"].values() for it in xs)
    check("a sheet item that is a board exemplar is refused",
          raises(lambda: SH.check_not_exemplars([{"unit_id": ex_one["id"]}], plans), SheetError)
          and SH.check_not_exemplars([{"unit_id": "b:not_an_exemplar#0"}], plans) is None)
    ex_item = next(it for xs in plans["B2"]["exemplars"].values() for it in xs)
    shared = dict(Q.item_keys(ex_item, independent=True), lab=domain.reference_lab())
    check("board_for: an item sharing every board's material is refused (reference lab, and an exemplar's "
          "observation)", raises(lambda: SH.board_for(shared, domain, plans), DisjointnessError))
    same_obs = Q.item_keys(dict(ex_item, id=ex_item["id"].rsplit("/", 1)[0] + "/2"), independent=True)
    check("a second photo of an exemplar's observation shares with the board that shows it",
          not Q.disjoint(same_obs, plans["B2"]["material"]))
    check("board_for: an independent-lab pool item goes to B1",
          SH.board_for({"source": FX.AN, "lab": "src:%s" % FX.AN, "near_dup3": "n:x", "provenance": "prov:x"},
                       domain, plans) == "B1")
    del kt_all


def FX_sha(p):
    from weed_optimizer_framework.tools.inc import common as C
    return C.sha256_file(p)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "-"))
    sys.exit(1 if FAILURES else 0)
