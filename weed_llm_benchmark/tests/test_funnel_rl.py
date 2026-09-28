#!/usr/bin/env python3
"""funnel/rl.py (runner docs/FUNNEL_AUDIT_RUNNER.md §4.11, §5.4.2; contract
docs/FUNNEL_AUDIT.md §4.3, DEC-1, DEC-2).

  * parse_item accepts "7, YES", "7,yes", "<think>...</think>\\n7, NA" and a
    JSON object, and refuses "seven", "27, YES" with 25 options, a missing
    box answer and a JSON option out of range;
  * parse_sheet maps panel positions (lines and JSON), leaves out positions
    answered twice differently and positions not on the sheet;
  * the ollama client: a fake transport answering garbage twice then a valid
    line succeeds on attempt 3 with the earlier replies kept; three garbage
    replies are recorded as unparsed with their raw text; an HTTP error
    counts as an attempt; check_vision refuses a model without vision and one
    whose capabilities are not reported; the request carries base64 images,
    temperature 0 and a seed per item and attempt; run_rl_b refuses a client
    whose model is not the config's and records the model digest; a rerun
    skips answered sheets;
  * validate_answers refuses, each on its own planted file: a sheet sha256
    that differs, a missing item, a doubled item, an option out of range, a
    backend other than the directory's, an RL-A file answering a sheet that
    shows evaluation pixels, an RL-A labeller of a judged model family;
  * ingest gives one gold row per item and labeller, fills the correct-answer
    columns for sentinels and identity items only, refuses everything when
    one file is invalid, and writes nothing for an item the key lacks.

Needs numpy and PIL (skips otherwise). No network: every HTTP call goes
through a fake transport.

Run:  python3 tests/test_funnel_rl.py
"""
import copy
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import funnel_labels_fixtures as FX  # noqa: E402

TMP = FX.setup("funnel_rl_")
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
    return None


class FakeOllama(object):
    """A fake /api transport. replies(prompt, images, seed, n) -> (status, text)."""

    def __init__(self, model, replies, caps=("completion", "vision"), digest="sha256:abc"):
        self.model, self.replies, self.caps, self.digest = model, replies, caps, digest
        self.calls = []

    def __call__(self, method, url, body, timeout):
        path = url.split("//", 1)[1].split("/", 1)[1]
        payload = json.loads(body.decode("utf-8")) if body else None
        self.calls.append((method, "/" + path, payload))
        if path == "api/show":
            out = {} if self.caps is None else {"capabilities": list(self.caps)}
            return 200, json.dumps(out).encode()
        if path == "api/tags":
            return 200, json.dumps({"models": [{"name": self.model, "digest": self.digest}]}).encode()
        if path == "api/chat":
            m = payload["messages"][0]
            status, text = self.replies(m["content"], m["images"], payload["options"]["seed"],
                                        len([c for c in self.calls if c[1] == "/api/chat"]))
            if status != 200:
                return status, b"server error"
            return 200, json.dumps({"message": {"role": "assistant", "content": text}}).encode()
        return 404, b"no such path"


def main():
    from weed_optimizer_framework.tools.inc import common as C
    from weed_optimizer_framework.tools.funnel import domain as D
    from weed_optimizer_framework.tools.funnel import rl as R
    from weed_optimizer_framework.tools.funnel import RLError, read_json, read_jsonl, read_csv, write_json_atomic

    # parsing
    check("parse_item '7, YES'", R.parse_item("7, YES", 25) == (7, "yes"))
    check("parse_item '7,yes'", R.parse_item("7,yes", 25) == (7, "yes"))
    check("parse_item after a think block", R.parse_item("<think>it is 3, no...</think>\n7, NA", 25) == (7, "na"))
    check("parse_item JSON object", R.parse_item('{"option": 4, "box": "NO"}', 25) == (4, "no"))
    check("parse_item fenced JSON", R.parse_item('```json\n{"option": 4, "box": "yes"}\n```', 25) == (4, "yes"))
    check("parse_item refuses 'seven'", R.parse_item("seven", 25) is None)
    check("parse_item refuses '27, YES' with 25 options", R.parse_item("27, YES", 25) is None)
    check("parse_item refuses an answer without the box part", R.parse_item("7", 25) is None)
    check("parse_item refuses a JSON option out of range", R.parse_item('{"option": 26, "box": "YES"}', 25) is None)
    check("parse_item refuses prose before the answer", R.parse_item("The answer is 7, YES", 25) is None)
    check("parse_pair", R.parse_pair("<think>x</think> 3", 4) == 3 and R.parse_pair("5", 4) is None)
    got = R.parse_sheet("1: 7, YES\n2: 3, no\n2: 4, NO\n3: 26, YES\n4 : 2 , NA\n16: 1, YES", range(1, 16), 25)
    check("parse_sheet maps positions; leaves out a doubled answer, an out-of-range option and a foreign "
          "position", got == {1: (7, "yes"), 4: (2, "na")}, got)
    got = R.parse_sheet('[{"panel": 1, "option": 2, "box": "YES"}, {"panel": 5, "option": 9, "box": "NA"}]',
                        [1, 5], 25)
    check("parse_sheet reads a JSON list", got == {1: (2, "yes"), 5: (9, "na")}, got)
    check("parse_sheet for a pair sheet", R.parse_sheet("1: 3\n2: 1", [1, 2], 4, pair=True) ==
          {1: (3, None), 2: (1, None)})

    missing = FX.have("numpy", "PIL")
    if missing:
        print("SKIP: %s not installed (the client and ingest parts need rendered sheets)" % ", ".join(missing))
        SKIPS.append("client+ingest")
        return
    fd = pathlib.Path(os.environ["INC_DIR"]) / "funnel"
    domain = D.load("weed")
    model = domain.raw["reference_labeller"]["backends"]["RL-B"]["model"]
    rla_model = domain.raw["reference_labeller"]["backends"]["RL-A"]["model"]
    n_opt = len(domain.options())

    # the client on one prompt
    seq = {"n": 0}

    def garbage_twice(prompt, images, seed, n):
        seq["n"] += 1
        return 200, ("I am not sure" if seq["n"] < 3 else "5, YES")
    fake = FakeOllama(model, garbage_twice)
    cl = R.OllamaClient("http://127.0.0.1:1", model, transport=fake)
    check("check_vision accepts a vision model", cl.check_vision() == ["completion", "vision"])
    check("model_digest from /api/tags", cl.model_digest() == "sha256:abc")
    check("check_vision refuses a model without vision",
          raises(lambda: R.OllamaClient("h:1", model, transport=FakeOllama(model, garbage_twice,
                                                                           caps=("completion",))).check_vision(),
                 RLError))
    check("check_vision refuses when capabilities are not reported",
          raises(lambda: R.OllamaClient("h:1", model, transport=FakeOllama(model, garbage_twice,
                                                                           caps=None)).check_vision(), RLError))

    world = FX.sheet_world(TMP, domain)
    from weed_optimizer_framework.tools.funnel import sheets as SH
    prereg = D.load_prereg(fd / "prereg_v1.json")
    SH.run(prereg, domain, fd, world["adapter"])
    key = {r["item_id"]: r for r in read_jsonl(fd / SH.KEY_DIR / "key.jsonl")}
    opts = domain.options()
    by_class = {o["class"]: o["n"] for o in opts if o["kind"] == "target"}
    by_taxon = {o["taxon"]: o["n"] for o in opts if o["kind"] == "attractor"}

    def truth_option(k):
        if k["truth_kind"] == "target":
            return by_class[int(k["truth"])]
        if k["truth_kind"] == "attractor":
            return by_taxon[k["truth_taxon"]]
        return n_opt - 3

    # RL-B: every sheet; item-level replies decided by the panel's item via the sheet json
    panels = {}
    for d in (SH.SHEETS_DIR, SH.CLUSTER_DIR):
        idx = read_json(fd / d / "index.json")
        for e in idx["sheets"]:
            doc = read_json(fd / d / ("%s.json" % e["sheet_id"]))
            for it in doc["items"]:
                panels[(e["sheet_id"], it["position"])] = it["item_id"]
    order = []
    for sid in sorted({s for s, _p in panels}):
        for pos in sorted(p for s, p in panels if s == sid):
            order.append(panels[(sid, pos)])
    state = {"i": 0, "tries": 0, "seeds": []}
    bad_item = order[1]

    def replies(prompt, images, seed, n):
        iid = order[state["i"]]
        state["seeds"].append((iid, seed))
        k = key[iid]
        if iid == bad_item:
            state["tries"] += 1
            if state["tries"] >= 3:
                state["i"] += 1
            return 200, "no idea"
        state["i"] += 1
        if k["pair_truth"] is not None or "Panel images A" in prompt:
            return 200, "1" if k.get("pair_truth") == "same" else "3"
        if k["is_sentinel"] or k["group"] == "identity":
            return 200, "%d, YES" % truth_option(k)
        return 200, "1, YES"
    fake = FakeOllama(model, replies)
    client = R.OllamaClient("http://127.0.0.1:1", model, transport=fake)
    check("run_rl_b refuses a client of another model",
          raises(lambda: R.run_rl_b(prereg, domain, fd, R.OllamaClient("h:1", "other:1b", transport=fake)),
                 RLError))
    R.run_rl_b(prereg, domain, fd, client)
    ans_dir = fd / "rl_answers" / "RL-B"
    files = sorted(p for p in ans_dir.glob("sheet_*.json"))
    check("run_rl_b: one answer file per sheet", len(files) == 6, len(files))
    all_ans = [a for p in files for a in read_json(p)["answers"]]
    bad = next(a for a in all_ans if a["item_id"] == bad_item)
    check("three garbage replies: unparsed, raw kept, 3 attempts",
          bad["status"] == "unparsed" and bad["option"] is None and bad["raw"] == "no idea"
          and bad["attempts"] == 3 and len(bad.get("raw_history", [])) == 2, bad)
    check("every other item parsed ok on its first attempt",
          all(a["status"] == "ok" and a["attempts"] == 1 for a in all_ans if a["item_id"] != bad_item))
    seeds_bad = [s for i, s in state["seeds"] if i == bad_item]
    check("the seed is stable_int(item) + attempt",
          seeds_bad == [C.stable_int("funnel/v1/rl-b/" + bad_item) + a for a in range(3)], seeds_bad)
    chat = [c for c in fake.calls if c[1] == "/api/chat"]
    mc_call = next(c for c in chat if "Panel images A" not in c[2]["messages"][0]["content"])
    pair_call = next(c for c in chat if "Panel images A" in c[2]["messages"][0]["content"])
    check("an item request carries the board and its panel (base64), temperature 0",
          len(mc_call[2]["messages"][0]["images"]) == 2 and mc_call[2]["options"]["temperature"] == 0
          and mc_call[2]["stream"] is False)
    check("a pair request carries its panel only", len(pair_call[2]["messages"][0]["images"]) == 1)
    check("the prompt is the config's with the options filled",
          mc_call[2]["messages"][0]["content"].startswith(domain.raw["reference_labeller"]["prompt"].split("{")[0])
          and "%d. %s" % (opts[0]["n"], opts[0]["text"]) in mc_call[2]["messages"][0]["content"])
    first = read_json(files[0])
    check("answer file records the model digest and labeller",
          first["model_digest"] == "sha256:abc" and first["labeller"] == "RL-B:%s" % model)
    n_before = len(fake.calls)
    R.run_rl_b(prereg, domain, fd, client)
    check("a rerun skips answered sheets", len([c for c in fake.calls[n_before:] if c[1] == "/api/chat"]) == 0)

    # a server that never answers: no answer file (resume would skip it), the run stops, a rerun resumes
    last = files[-1]
    saved_last = last.read_bytes()
    last.unlink()
    others = {p.name: p.read_bytes() for p in files[:-1]}
    down = FakeOllama(model, lambda prompt, images, seed, n: (500, ""))
    err = raises(lambda: R.run_rl_b(prereg, domain, fd, R.OllamaClient("h:1", model, transport=down)), RLError)
    check("every request failing stops rl-b (a server failure is not an unparsed answer)",
          err and "every one of 3 request(s) failed" in err and not last.exists(), err)
    check("... and the sheets answered before are kept byte for byte",
          all((ans_dir / n).read_bytes() == b for n, b in others.items()))

    def fine(prompt, images, seed, n):
        return 200, ("1" if "Panel images A" in prompt else "1, YES")
    R.run_rl_b(prereg, domain, fd, R.OllamaClient("h:1", model, transport=FakeOllama(model, fine)))
    check("a rerun after the outage writes the missing sheet", last.exists()
          and all(a["status"] == "ok" for a in read_json(last)["answers"]))
    last.write_bytes(saved_last)

    # an HTTP error counts as an attempt
    seq2 = {"n": 0}

    def flaky(prompt, images, seed, n):
        seq2["n"] += 1
        return (500, "") if seq2["n"] == 1 else (200, "2, NO")
    c2 = R.OllamaClient("h:1", model, transport=FakeOllama(model, flaky))
    try:
        c2.ask("p", [b"x"], 1)
        http_err = False
    except RLError:
        http_err = True
    check("an HTTP error raises RLError (retried by run_rl_b)", http_err)

    # RL-A answers the pool sheets through answers_from_text
    rla_dir = fd / "rl_answers" / "RL-A"
    pool_idx = read_json(fd / SH.SHEETS_DIR / "index.json")
    for e in pool_idx["sheets"]:
        sp = fd / SH.SHEETS_DIR / ("%s.json" % e["sheet_id"])
        doc = read_json(sp)
        lines = []
        for it in doc["items"]:
            k = key[it["item_id"]]
            opt = truth_option(k) if (k["is_sentinel"] or k["group"] == "identity") else 1
            lines.append("%d: %d, YES" % (it["position"], opt))
        R.answers_from_text(sp, "\n".join(lines), "RL-A", "RL-A:%s" % rla_model, out_dir=rla_dir, domain=domain)
    gold = R.ingest(prereg, domain, fd, [ans_dir, rla_dir])
    _h, grows = read_csv(fd / "gold_v1.csv")
    n_items = len(key)
    n_pool = sum(e["n_items"] for e in pool_idx["sheets"])
    check("ingest: one gold row per item and labeller", len(grows) == n_items + n_pool == gold["rows"],
          (len(grows), n_items, n_pool))
    check("gold rows name their labeller and backend",
          {r["backend"] for r in grows} == {"RL-A", "RL-B"}
          and all(r["labeller"] == "%s:%s" % (r["backend"], model if r["backend"] == "RL-B" else rla_model)
                  for r in grows))
    q = [r for r in grows if r["is_sentinel"] == "1" or r["group"] == "identity"]
    nq = [r for r in grows if not (r["is_sentinel"] == "1" or r["group"] == "identity")]
    check("correct-answer columns filled for sentinels and identity items (pairs excepted)",
          all(r["correct_species"] in ("0", "1") for r in q if not r["unit_id"].startswith("p:")))
    check("correct-answer columns empty for everything else", all(r["correct_species"] == "" for r in nq))
    wrong = [(r["unit_id"], r["answer"], r["truth_kind"], r["truth_taxon"], r["correct_species"],
              r["correct_genus"], r["correct_plant"]) for r in q if not r["unit_id"].startswith("p:")
             and r["answer"] != "unparsed"
             and not (r["correct_species"] == "1" and r["correct_genus"] == "1" and r["correct_plant"] == "1")]
    check("sentinels answered with their truth are correct at every level", not wrong, wrong[:4])
    unp = [r for r in grows if r["item_id"] == bad_item and r["backend"] == "RL-B"]
    check("an unparsed answer is ingested as unparsed", unp and unp[0]["answer"] == "unparsed")
    tail = [r for r in grows if r["answer"] in ("same", "different")]
    check("pair answers map to the pair vocabulary", tail and all(r["answer_level"] == "none" for r in tail))

    # validation refusals, each planted
    sheet_ids = [e["sheet_id"] for e in pool_idx["sheets"]]
    sp = fd / SH.SHEETS_DIR / ("%s.json" % sheet_ids[0])
    good = read_json(rla_dir / ("%s.json" % sheet_ids[0]))
    check("a good RL-A file validates", R.validate_answers(rla_dir / ("%s.json" % sheet_ids[0]), sp, "RL-A",
                                                          domain=domain) == [])

    def planted(mut, backend="RL-A", sheet=sp, name=None):
        d = TMP / "planted" / backend
        rec = copy.deepcopy(good)
        mut(rec)
        p = d / ("%s.json" % (name or rec["sheet_id"]))
        write_json_atomic(p, rec)
        return R.validate_answers(p, sheet, backend, domain=domain)
    check("refused: sheet sha256 differs", planted(lambda r: r.update(sheet_sha256="0" * 64)))
    check("refused: an item missing", planted(lambda r: r["answers"].pop()))
    check("refused: an item doubled", planted(lambda r: r["answers"].append(dict(r["answers"][0]))))
    check("refused: an option out of range", planted(lambda r: r["answers"][0].update(option=n_opt + 1)))
    check("refused: backend not the directory's", planted(lambda r: None, backend="RL-B"))
    probs = planted(lambda r: r.update(board_sha256="0" * 64))
    check("refused: a board sha256 that is not the sheet's board", any("board_sha256" in p for p in probs), probs)
    probs = planted(lambda r: r.update(labeller="RL-A:another-model"))
    check("refused: an RL-A labeller that is not the configured model",
          any("is not the configured" in p for p in probs), probs)
    rb_good = read_json(ans_dir / ("%s.json" % sheet_ids[0]))
    d_b = TMP / "planted_b" / "RL-B"
    write_json_atomic(d_b / ("%s.json" % sheet_ids[0]), dict(rb_good, model_digest=None))
    probs = R.validate_answers(d_b / ("%s.json" % sheet_ids[0]), sp, "RL-B", domain=domain)
    check("refused: an RL-B file without the model digest", probs and all("model digest" in p for p in probs), probs)
    pair_sid = read_json(fd / SH.CLUSTER_DIR / "index.json")["sheets"][0]["sheet_id"]
    pair_sheet = fd / SH.CLUSTER_DIR / ("%s.json" % pair_sid)
    rb_pair = read_json(ans_dir / ("%s.json" % pair_sid))

    def as_rla(r):
        r.clear()
        r.update(copy.deepcopy(rb_pair))
        r.update(backend="RL-A", labeller="RL-A:%s" % rla_model)
    probs = planted(as_rla, sheet=pair_sheet, name=pair_sid)
    check("refused: an RL-A file for a sheet with evaluation pixels (DEC-2)",
          any("evaluation pixels" in p for p in probs), probs)
    probs = planted(lambda r: r.update(labeller="RL-A:qwen2.5-vl:7b"))
    check("refused: an RL-A labeller of a judged family", any("judged model family" in p for p in probs), probs)
    rb_ok = R.validate_answers(ans_dir / ("%s.json" % pair_sid), pair_sheet, "RL-B", domain=domain)
    check("RL-B may answer the pair sheet", rb_ok == [], rb_ok)

    # ingest refuses when one file is invalid
    bad_p = rla_dir / ("%s.json" % sheet_ids[1])
    saved = bad_p.read_bytes()
    rec = read_json(bad_p)
    rec["answers"].pop()
    write_json_atomic(bad_p, rec)
    check("ingest refuses everything when one answer file is invalid",
          raises(lambda: R.ingest(prereg, domain, fd, [ans_dir, rla_dir]), RLError))
    bad_p.write_bytes(saved)

    # the sample (the items' disjointness keys) is required and must be the locked one
    smp = fd / "sample_v1.csv"
    smp_bytes = smp.read_bytes()
    smp.unlink()
    check("ingest refuses without the sample (no silent gold rows without disjointness keys)",
          raises(lambda: R.ingest(prereg, domain, fd, [ans_dir, rla_dir]), RLError))
    assert b",0.5," in smp_bytes
    smp.write_bytes(smp_bytes.replace(b",0.5,", b",0.6,", 1))       # a parseable sample that is not the locked one
    from weed_optimizer_framework.tools.funnel import StaleInput
    check("ingest refuses a sample that is not the locked one",
          raises(lambda: R.ingest(prereg, domain, fd, [ans_dir, rla_dir]), StaleInput))
    smp.write_bytes(smp_bytes)
    check("gold rows carry the sample's disjointness keys",
          all(r["near_dup3"] and r["provenance"] for r in grows if r["group"] in ("sentinel", "identity")))

    # an item the key lacks: the key is rewritten (and both indexes updated) without one item
    kp = fd / SH.KEY_DIR / "key.jsonl"
    rows = read_jsonl(kp)
    dropped = rows[0]["item_id"]
    from weed_optimizer_framework.tools.funnel import write_jsonl_atomic
    ksha = write_jsonl_atomic(kp, rows[1:])
    for d in (SH.SHEETS_DIR, SH.CLUSTER_DIR):
        ip = fd / d / "index.json"
        idx = read_json(ip)
        idx["key_sha256"] = ksha
        write_json_atomic(ip, idx)
    gold2 = R.ingest(prereg, domain, fd, [ans_dir, rla_dir])
    _h, g2 = read_csv(fd / "gold_v1.csv")
    check("ingest writes nothing for an item missing from the key",
          dropped not in {r["item_id"] for r in g2} and dropped in gold2["not_in_key"])
    check("gold_v1.json records the gold sha256 and the key sha256",
          gold2["gold_sha256"] == C.sha256_file(fd / "gold_v1.csv") and gold2["key_sha256"] == ksha)


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(SKIPS) or "-"))
    sys.exit(1 if FAILURES else 0)
