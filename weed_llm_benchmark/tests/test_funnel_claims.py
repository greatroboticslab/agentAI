#!/usr/bin/env python3
"""The claims register (contract §8.3, §8.6; runner §4.17, §5.1.7).

Pinned:
  * C1 and C2 as runner §5.5.10 files them (human-transcribed; the RESEARCH_LOG
    line and the poster figure are cited) validate, and their cites resolve
    in this repository (C1 was filed from line 154 of RESEARCH_LOG.md at 8b50e52; its text, the
    poster figure data holds the C2 pointer);
  * every allowed transition works for each allowed actor kind, and every
    other (status pair, actor) is refused;
  * the adversary needs a valid reply with a surviving counter-argument from
    another model family than the planner's resolved one: a same-family
    reply, an invalid reply and a reply with nothing surviving are refused;
  * the autopilot needs a valid audit on a matching ledger with D18 silent,
    and a cite carrying the audit's sha256;
  * a person signs human:<actor>;
  * history is append-only: save() refuses a rewritten entry, a removed
    claim and a changed filed field;
  * open_negative lists open and challenged scarcity/negative claims only;
  * the CLI files, lists and moves claims with exit code 2 on a refusal;
  * the module imports with the standard library only (checked in
    test_funnel_init.py; here: no numpy in its import lines).

Run:  python3 tests/test_funnel_claims.py
"""
import copy
import io
import json
import pathlib
import re
import sys
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import funnel_stats_world as W  # noqa: E402

TMP = W.setup("funnel_claims_")
check, raises = W.check, W.raises

from weed_optimizer_framework.tools import funnel as F  # noqa: E402
from weed_optimizer_framework.tools.funnel import claims as CL  # noqa: E402

GIT = W.GIT_ROOT
C1_TEXT = ("The harvested increments do not improve the twelve target species. They are mostly other plants plus "
           "about 50 target-species boxes each.")
C2_TEXT = "web harvest at this scale supplies volume, not usable supervision"


def cite(artifact, value, line=None, pointer=None):
    return {"artifact": artifact, "line": line, "pointer": pointer, "value": value}


print("seed claims C1 and C2")
log = (GIT / "RESEARCH_LOG.md").read_text(encoding="utf-8").splitlines()
# C1 was filed from line 154 (commit 8b50e52); entries are added above it, so it is found by its text
hits = [i for i, ln in enumerate(log) if "harvested increments do not improve" in ln.lower()]
line154 = log[hits[0]] if len(hits) == 1 else ""
check("RESEARCH_LOG.md carries C1's sentence once", len(hits) == 1, hits)
fig = json.loads((GIT / "docs" / "poster" / "figures_data.json").read_text(encoding="utf-8"))
reading = (fig.get("s1_gate_verdict_2026_08_25") or {}).get("reading")
check("the poster figure carries C2's pointer", isinstance(reading, str) and "supervision" in reading, reading)
reg = CL.new_register()
c1 = CL.file_claim(reg, C1_TEXT, "negative", "realloop_v1 increments on dev", "human-transcribed",
                   [cite("RESEARCH_LOG.md", line154, line=154),
                    cite("realloop_v1/report.json", {"note": "agreement"}, pointer="/agreement/full")])
c2 = CL.file_claim(reg, C2_TEXT, "scarcity", "web harvest", "human-transcribed",
                   [cite("docs/poster/figures_data.json", reading, pointer="/s1_gate_verdict_2026_08_25/reading")])
check("ids C1, C2", (c1, c2) == ("C1", "C2"))
check("seed register validates", CL.validate(reg) == [], CL.validate(reg))
check("filed open, with a filing history entry", all(c["status"] == "open" and c["history"][0]["to"] == "open"
                                                      and c["history"][0]["from"] is None for c in reg["claims"]))
check("open_negative lists both", [c["id"] for c in CL.open_negative(reg)] == ["C1", "C2"])
path = TMP / "campaign" / "claims.json"
CL.save(path, reg)
check("save/load round trip", CL.load(path) == json.loads(F.json_text(reg)))
check("a missing register is empty", CL.load(TMP / "none.json") == CL.new_register())

print("filing refusals")
for name, args in (("bad polarity", (C1_TEXT, "gloomy", "s", "human-transcribed", [])),
                   ("empty text", (" ", "negative", "s", "human-transcribed", [])),
                   ("bad made_by", (C1_TEXT, "negative", "s", "somebody", [])),
                   ("bad cite", (C1_TEXT, "negative", "s", "human-transcribed", [{"artifact": "x"}])),
                   ("cite without line or pointer", (C1_TEXT, "negative", "s", "human-transcribed",
                                                     [cite("x", 1)]))):
    check("file_claim refuses %s" % name, raises(lambda a=args: CL.file_claim(copy.deepcopy(reg), *a),
                                                 F.ClaimsError))

print("model family")
check("vllm:glm-4.7-flash -> glm", CL.model_family("vllm:glm-4.7-flash") == "glm")
check("ollama:qwen3.8:27b -> qwen", CL.model_family("ollama:qwen3.8:27b") == "qwen")
check("ollama:gemma4 -> gemma", CL.model_family("ollama:gemma4") == "gemma")
check("claude-opus-5-5 -> claude", CL.model_family("claude-opus-5-5") == "claude")

AUDIT_SHA = W.sha("audit")
PROOF = {"adversary": {"valid": True, "surviving": 3, "model": "vllm:glm-4.7-flash",
                       "planner_model": "ollama:qwen3.8:27b", "same_family": False},
         "autopilot": {"audit_sha256": AUDIT_SHA, "valid": True, "fingerprint_match": True, "d18_fired": False},
         "person": None}
BY = {"adversary": "tier2:vllm:glm-4.7-flash", "autopilot": "round-scheduler:inc-autopilot",
      "person": "human:owner"}
CITES = {"adversary": [cite("step1/calibration.json", 0.9457, pointer="/x")],
         "autopilot": [cite("funnel/audit_v1.json", AUDIT_SHA, pointer="")],
         "person": []}


def at(status):
    """A one-claim register whose claim sits at `status` (moved by a person
    or the adversary along allowed transitions)."""
    r = CL.new_register()
    CL.file_claim(r, C1_TEXT, "negative", "s", "human-transcribed", [])
    path_ = {"open": [], "challenged": ["challenged"], "tested_survives": ["challenged", "tested_survives"],
             "refuted": ["challenged", "refuted"], "accepted_open": ["accepted_open"]}[status]
    for to in path_:
        CL.transition(r, "C1", to, "human:owner", "setup", [], "person")
    return r


print("transitions")
for (frm, to), actors in sorted(CL.TRANSITIONS.items()):
    for kind in CL.ACTOR_KINDS:
        r = at(frm)
        allowed = kind in actors

        def go(r=r, to=to, kind=kind):
            return CL.transition(r, "C1", to, BY[kind], "reason", CITES[kind], kind, PROOF[kind])
        if allowed:
            cl = go()
            check("%s -> %s by %s" % (frm, to, kind), cl["status"] == to and cl["history"][-1]["from"] == frm
                  and cl["history"][-1]["by"] == BY[kind])
            check("register valid after %s -> %s" % (frm, to), CL.validate(r) == [])
        else:
            check("%s -> %s refused for %s" % (frm, to, kind), raises(go, F.ClaimsError))
n_bad = 0
for frm in CL.STATUSES:
    for to in CL.STATUSES:
        if (frm, to) in CL.TRANSITIONS or frm == to:
            continue
        r = at(frm)
        if not raises(lambda r=r, to=to: CL.transition(r, "C1", to, "human:owner", "x", [], "person"),
                      F.ClaimsError):
            n_bad += 1
check("every transition outside TRANSITIONS is refused, even for a person", n_bad == 0, n_bad)

print("adversary rules")
r = at("open")
for name, proof in (("same family", dict(PROOF["adversary"], model="ollama:qwen3.8:9b")),
                    ("same family by flag", dict(PROOF["adversary"], same_family=True)),
                    ("invalid reply", dict(PROOF["adversary"], valid=False)),
                    ("nothing surviving", dict(PROOF["adversary"], surviving=0)),
                    ("no planner model", dict(PROOF["adversary"], planner_model=None)),
                    ("no proof", None)):
    check("adversary refused: %s" % name,
          raises(lambda p=proof: CL.transition(r, "C1", "challenged", BY["adversary"], "x", CITES["adversary"],
                                               "adversary", p), F.ClaimsError))
check("adversary without cites refused",
      raises(lambda: CL.transition(r, "C1", "challenged", BY["adversary"], "x", [], "adversary",
                                   PROOF["adversary"]), F.ClaimsError))
check("the claim did not move", r["claims"][0]["status"] == "open" and len(r["claims"][0]["history"]) == 1)

print("autopilot rules")
r = at("challenged")
for name, proof, cites in (("invalid audit", dict(PROOF["autopilot"], valid=False), CITES["autopilot"]),
                           ("fingerprint mismatch", dict(PROOF["autopilot"], fingerprint_match=False),
                            CITES["autopilot"]),
                           ("D18 fired", dict(PROOF["autopilot"], d18_fired=True), CITES["autopilot"]),
                           ("no cite of the audit sha", PROOF["autopilot"], [cite("x", "y", pointer="")]),
                           ("not a sha", dict(PROOF["autopilot"], audit_sha256="abc"), CITES["autopilot"])):
    check("autopilot refused: %s" % name,
          raises(lambda p=proof, c=cites: CL.transition(r, "C1", "tested_survives", BY["autopilot"], "x", c,
                                                        "autopilot", p), F.ClaimsError))
check("a person must sign human:<actor>",
      raises(lambda: CL.transition(at("challenged"), "C1", "refuted", "owner", "x", [], "person"), F.ClaimsError))

print("append-only history")
r = at("challenged")
p2 = TMP / "c2.json"
CL.save(p2, r)
r2 = copy.deepcopy(r)
CL.transition(r2, "C1", "tested_survives", "human:owner", "audited", [], "person")
CL.save(p2, r2)
check("appending a transition saves", CL.load(p2)["claims"][0]["status"] == "tested_survives")
r3 = copy.deepcopy(r2)
r3["claims"][0]["history"][1]["reason"] = "rewritten"
check("a rewritten history entry is refused", raises(lambda: CL.save(p2, r3), F.ClaimsError))
r4 = copy.deepcopy(r2)
r4["claims"][0]["text"] = "something else"
check("a changed filed field is refused", raises(lambda: CL.save(p2, r4), F.ClaimsError))
r5 = CL.new_register()
check("a removed claim is refused", raises(lambda: CL.save(p2, r5), F.ClaimsError))
r6 = copy.deepcopy(r2)
r6["claims"][0]["status"] = "refuted"
check("a status that disagrees with the history is invalid", CL.validate(r6) != [])
check("open_negative excludes tested_survives", CL.open_negative(r2) == [])
pos = CL.new_register()
CL.file_claim(pos, "recovered data helps", "positive", "s", "human:owner", [])
check("open_negative excludes positive claims", CL.open_negative(pos) == [])

print("CLI")
cp = TMP / "cli.json"


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rc = CL.main(argv)
    return rc, out.getvalue(), err.getvalue()


rc, out, _ = run(["file", "--path", str(cp), "--text", C2_TEXT, "--polarity", "scarcity", "--scope", "web",
                  "--made-by", "human-transcribed", "--cite", json.dumps(cite("f.json", 1, pointer="/a"))])
check("CLI file", rc == 0 and "C1" in out)
rc, out, _ = run(["list", "--path", str(cp)])
check("CLI list", rc == 0 and out.startswith("C1\topen\tscarcity"))
rc, _o, err = run(["transition", "--path", str(cp), "--id", "C1", "--to", "refuted", "--by", "human:owner",
                   "--reason", "x", "--actor-kind", "person"])
check("CLI refusal exits 2", rc == 2 and "refused" in err)
rc, out, _ = run(["transition", "--path", str(cp), "--id", "C1", "--to", "challenged", "--by", "tier2:vllm:glm",
                  "--reason", "x", "--actor-kind", "adversary", "--cite", json.dumps(cite("a", 1, pointer="/")),
                  "--proof", json.dumps(PROOF["adversary"])])
check("CLI adversary transition", rc == 0 and CL.load(cp)["claims"][0]["status"] == "challenged")

print("standard library only")
src = pathlib.Path(CL.__file__).read_text()
imports = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
check("no third-party import", not any(re.search(r"\b(numpy|scipy|torch|sklearn|PIL)\b", ln) for ln in imports),
      imports)

W.finish()
