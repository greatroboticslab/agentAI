"""Protocol v3's gate: the pinned gate's decision with only the species
guard's tolerance replaced, per species (docs/CONTINUOUS_LOOP.md §2.6 L-3;
docs/INCREMENTAL_PROTOCOL.md, "Protocol v3").

    python -m weed_optimizer_framework.tools.inc2.gate3 decide --exp EXP [--no-verify] [--out PATH]
    python -m weed_optimizer_framework.tools.inc2.gate3 show   --exp EXP

The rule (pre-registered, L-3). For each of the 12 species s,
    tol_s = max(0.03, 1.96 x SE_s),
SE_s being the image-bootstrap standard error of the incumbent's dev AP for s:
1,000 resamples of the dev images, seed stable_int("inc2/v3/species_se"),
computed by inc2.scorer_sidecar from the incumbent's own dev predictions and
recorded in its runs/<incumbent>/scores/dev.sidecar.json. The species guard
passes s when mean(cand AP_s) - mean(null AP_s) >= -tol_s. Nothing else
changes: P_data, P_recipe, the regression guard, the flips guard, their
thresholds and the verdict rule (REJECT on a failed guard, else ACCEPT at
P_data >= p_accept, REJECT at P_data <= p_reject, HOLD between) are the
pinned gate's (inc/gate.py, which stays pinned), with the experiment's own
pinned GateConfig (flips_mode net for every v2 chain).

How a step is re-decided. The input is the pinned driver's ledger entry of
that step ("gate/<chain>/<k>"), whose decision the pinned gate made:
  * verify (the default): every score file the entry lists (the incumbent,
    each cand, each null) is read and must hash to the sha256 the entry
    recorded; gate.decide is run again on them with the entry's own config,
    and its decision must equal the recorded one exactly (the canonical JSON
    of both). A step the pinned gate does not reproduce is refused;
  * the incumbent's sidecar must name the incumbent's weights (the decision's
    weights.inc) and the score's stamps (exam dev, scorer, manifest, key
    order, n_images), use the pre-registered seed text and resample count,
    and hold a finite SE for every species. Anything else is refused;
  * the species guard is recomputed from the per-species deltas the pinned
    guard recorded (the same means of the same per-class APs), with tol_s in
    place of the pinned max(0.03, 3 sd_species(null)); the verdict follows;
    the attribution keeps every recorded field, with blame (recipe / data at
    a REJECT, as gate._attribute sets it) and species_failed following the
    new verdict and guard.
The truth arm's species guard is replaced the same way (truth3): tol_s from
the sidecar of the "without" arm's first run, P and the rest as recorded.

decide_experiment(exp) re-decides every gate and truth entry of a chain
experiment and writes INC_DIR/<exp>/gate3.json atomically (format
inc2-gate3/1): per step the v3 decision, the pinned verdict, whether it
changed, and the sha256 of every input (the ledger, each entry, each score
file, each sidecar). A step whose v3 decision cannot be made (no sidecar, a
refused input) is recorded with status 'unavailable', the reason, v3_applied
false, and a commit_verdict that never accepts: the pinned verdict, except
that a pinned ACCEPT becomes HOLD. Nothing is accepted on a rule that could
not be applied (the v3 tolerance can be stricter than the pinned one for a
species whose null spread is wide), and a HOLD returns the increment to the
queue once instead of admitting it. A pinned REJECT or HOLD stands (v3 only
changes the species tolerance, so it cannot turn those into an ACCEPT except
on a species-only failure, which then returns once as 'species'). A truth
entry that cannot be re-decided keeps its pinned verdict (commit_rule says
so).

How a stream commit uses it (group E, inc2.stream commit; docs/INCREMENTAL_
PROTOCOL_RUNNER.md, "Protocol v3"): after a segment <sid>_sNNN is done, run
decide_experiment (or this CLI) and read gate3.json. For the chosen chain,
an increment is ACCEPTed into P_s when its step's commit_verdict is ACCEPT;
the §3.5 disposition of a REJECT reads that step's v3 failed_guards (data =
p_data_le_p_reject unless the truth record says helps; then species, flips,
recipe = regression only); truth dispositions read the truth section's
commit_verdict. The pinned chain inside the segment is not re-run: a step v3
accepts that the pinned gate rejected joins P_s on its own step's evidence
(cand on pool + D_k against null on the pool, from the same incumbent), and
the next segment cold-trains on P_s.

No Ultralytics, torch or numpy here: a v3 decision is re-derived from the
ledger, the score files and the sidecar JSON on any machine.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import sys
from pathlib import Path

from ..inc import common as C
from ..inc import gate as G

FORMAT = "inc2-gate3/1"
PROTOCOL = "v3"
OUT_NAME = "gate3.json"
SIDECAR_FORMAT = "inc2-scorer-sidecar/1"          # inc2.scorer_sidecar.FORMAT (no numpy import here)
SIDECAR_NAME = "dev.sidecar.json"
AVAILABLE, UNAVAILABLE = "decided", "unavailable"
SIDECAR_STAMPS = ("exam", "scorer_sha256", "manifest_sha256", "key_order_sha256", "n_images")


@dataclasses.dataclass(frozen=True)
class Gate3Config:
    """L-3's constants; the defaults are the pre-registration."""
    species_floor: float = 0.03
    species_z: float = 1.96
    resamples: int = 1000
    seed_text: str = "inc2/v3/species_se"


class Gate3Error(ValueError):
    """A v3 decision cannot be made on these inputs."""


def log(msg):
    print("[inc2.gate3] %s" % msg, flush=True)


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".%s.%d.tmp" % (path.name, os.getpid()))
    try:
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True, allow_nan=False)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return C.sha256_file(path)


# ------------------------------------------------------------------ sidecar
def load_sidecar(sidecar):
    """(dict, {"path", "sha256"} or None) from a path or an already-read dict."""
    if isinstance(sidecar, dict):
        return sidecar, None
    p = Path(sidecar)
    try:
        with open(p, "rb") as fh:
            raw = fh.read()
    except OSError as e:
        raise Gate3Error("no sidecar at %s (%s)" % (p, e))
    try:
        d = json.loads(raw.decode("utf-8"))
    except ValueError as e:
        raise Gate3Error("sidecar %s is not JSON (%s)" % (p, e))
    return d, {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest()}


def species_tolerances(sidecar, cfg=Gate3Config(), weights_sha256=None, stamps=None, require_production=True):
    """{species: {"se", "tolerance", "n_valid", "n_gt", "ap"}} from a sidecar
    record, after checking it is the one the rule needs (module docstring)."""
    if not isinstance(sidecar, dict) or sidecar.get("format") != SIDECAR_FORMAT:
        raise Gate3Error("not an inc2 scorer sidecar (format %r)" % (sidecar or {}).get("format"))
    if sidecar.get("exam") != G.DECISION_EXAM:
        raise Gate3Error("the sidecar is on %r; the species guard reads %r" % (sidecar.get("exam"), G.DECISION_EXAM))
    if weights_sha256 is not None and sidecar.get("weights_sha256") != weights_sha256:
        raise Gate3Error("the sidecar is of weights %s, the incumbent is %s"
                         % (str(sidecar.get("weights_sha256"))[:12], str(weights_sha256)[:12]))
    rec_stamps = ((sidecar.get("score") or {}).get("stamps") or {})
    for k, v in (stamps or {}).items():
        if k in SIDECAR_STAMPS and rec_stamps.get(k) != v:
            raise Gate3Error("the sidecar's score has %s %s, the decision's inputs %s"
                             % (k, str(rec_stamps.get(k))[:16], str(v)[:16]))
    if require_production and (sidecar.get("score") or {}).get("production") is not True:
        raise Gate3Error("the sidecar was made from a test-mode score; a production decision needs a production one")
    se = sidecar.get("species_se") or {}
    if se.get("seed_text") != cfg.seed_text or se.get("resamples") != cfg.resamples:
        raise Gate3Error("the sidecar's bootstrap is seed %r x %r resamples; L-3 pre-registers %r x %d"
                         % (se.get("seed_text"), se.get("resamples"), cfg.seed_text, cfg.resamples))
    per = se.get("per_species") or {}
    out = {}
    for s in G.SPECIES:
        v = per.get(s)
        if not isinstance(v, dict) or not _finite(v.get("se")) or v["se"] < 0:
            raise Gate3Error("the sidecar has no finite SE for %s (%r)" % (s, v))
        if not isinstance(v.get("n_valid"), int) or v["n_valid"] < 2:
            raise Gate3Error("the sidecar's SE for %s rests on %r resamples" % (s, v.get("n_valid")))
        out[s] = {"se": float(v["se"]), "tolerance": max(cfg.species_floor, cfg.species_z * float(v["se"])),
                  "n_valid": v["n_valid"], "n_gt": v.get("n_gt"), "ap": v.get("ap")}
    return out


def species_guard(recorded, tolerances, a="cand", b="null"):
    """The v3 species guard from a pinned species guard record (its per-species
    deltas) and the per-species tolerances."""
    if not isinstance(recorded, dict) or not isinstance(recorded.get("per_species"), dict):
        raise Gate3Error("the recorded species guard has no per_species")
    per, failed = {}, []
    for s in G.SPECIES:
        r = recorded["per_species"].get(s)
        if not isinstance(r, dict) or not _finite(r.get("delta")):
            raise Gate3Error("the recorded species guard has no delta for %s" % s)
        tol = tolerances[s]["tolerance"]
        ok = G._ge(float(r["delta"]), -tol)
        per[s] = {"delta": r["delta"], "n_gt": r.get("n_gt"), "se": tolerances[s]["se"], "threshold": tol,
                  "pinned_threshold": r.get("threshold"), "pinned_sd": r.get("sd"), "pinned_passed": r.get("passed"),
                  "passed": ok}
        if not ok:
            failed.append(s)
    return {"passed": not failed, "arms": list(recorded.get("arms") or [a, b]), "rule": "v3",
            "checked": list(G.SPECIES), "failed": failed, "per_species": per,
            "min_species_gt": recorded.get("min_species_gt")}


def verdict_of(guards_pass, p_data, cfg):
    """The pinned gate's verdict rule (gate._verdict)."""
    if not guards_pass:
        return G.REJECT
    if G._ge(p_data, cfg.p_accept):
        return G.ACCEPT
    if G._le(p_data, cfg.p_reject):
        return G.REJECT
    return G.HOLD


def truth_verdict_of(species_pass, p, cfg):
    """The pinned truth rule (gate.truth_detail)."""
    if not species_pass or G._le(p, cfg.p_reject):
        return G.HURTS
    if G._ge(p, cfg.p_accept):
        return G.HELPS
    return G.NEUTRAL


# ------------------------------------------------------------------ inputs
def _read_scores(items, what):
    """Score dicts from [{run_id, path, sha256}], each checked against its
    recorded sha256."""
    out = []
    for it in items:
        try:
            with open(it["path"], "rb") as fh:
                raw = fh.read()
        except (OSError, KeyError, TypeError) as e:
            raise Gate3Error("%s score %s cannot be read (%s)" % (what, (it or {}).get("path"), e))
        if hashlib.sha256(raw).hexdigest() != it.get("sha256"):
            raise Gate3Error("%s score %s no longer hashes to the sha256 the ledger recorded" % (what, it["path"]))
        out.append(json.loads(raw.decode("utf-8")))
    return out


def pinned_config(decision):
    try:
        return G.GateConfig(**decision["config"])
    except (KeyError, TypeError, ValueError) as e:
        raise Gate3Error("the recorded decision's config is not a GateConfig: %s" % e)


def rederive(entry):
    """Re-run the pinned gate on the entry's recorded score files (checked
    against their sha256) and require the recorded decision back exactly."""
    d = entry["decision"]
    cfg = pinned_config(d)
    ins = entry.get("inputs") or {}
    (inc,) = _read_scores([ins["inc"]], "incumbent")
    cands = _read_scores(ins["cand"], "cand")
    nulls = _read_scores(ins["null"], "null")
    try:
        again = G.decide(inc, cands, nulls, cfg).to_dict()
    except (ValueError, KeyError, TypeError) as e:
        raise Gate3Error("the pinned gate refuses the recorded inputs: %s: %s" % (type(e).__name__, e))
    if _canon(again) != _canon(d):
        diff = sorted(k for k in set(again) | set(d) if _canon(again.get(k)) != _canon(d.get(k)))
        raise Gate3Error("the pinned gate does not reproduce the recorded decision (fields %s)" % diff)
    return True


def rederive_truth(entry, cfg):
    ins = entry.get("inputs") or {}
    w = _read_scores(ins["with"], "truth with")
    wo = _read_scores(ins["without"], "truth without")
    try:
        again = G.truth_detail(w, wo, cfg)
    except (ValueError, KeyError, TypeError) as e:
        raise Gate3Error("the pinned truth rule refuses the recorded inputs: %s: %s" % (type(e).__name__, e))
    if _canon(again) != _canon(entry["detail"]):
        raise Gate3Error("the pinned truth rule does not reproduce the recorded detail")
    return True


# ---------------------------------------------------------------- decisions
def decide(entry, sidecar, cfg=Gate3Config(), verify=True):
    """The v3 decision of one pinned gate ledger entry (module docstring).
    sidecar is the incumbent's sidecar (a path or its dict)."""
    if not isinstance(entry, dict) or entry.get("type") != "gate" or "decision" not in entry:
        raise Gate3Error("not a pinned gate ledger entry")
    d = entry["decision"]
    pcfg = pinned_config(d)
    if verify:
        rederive(entry)
    sc, sc_file = load_sidecar(sidecar)
    tols = species_tolerances(sc, cfg, weights_sha256=(d.get("weights") or {}).get("inc"),
                              stamps=d.get("stamps") or {}, require_production=pcfg.require_production)
    guards = {"regression": d["guards"]["regression"], "species": species_guard(d["guards"]["species"], tols),
              "flips": d["guards"]["flips"]}
    failed = [g for g in ("regression", "species", "flips") if not guards[g]["passed"]]
    verdict = verdict_of(not failed, d["p_data"], pcfg)
    attr = dict(d.get("attribution") or {})
    attr["blame"] = ("recipe" if attr.get("recipe_flag") else "data") if verdict == G.REJECT else None
    attr["species_failed"] = list(guards["species"]["failed"])
    pinned_verdict = d["verdict"]
    reason = "%s (pinned %s): P_data=%.3f; %s" % (
        verdict, pinned_verdict, d["p_data"],
        ("guard failed: " + ", ".join(failed)) if failed else "guards pass")
    if guards["species"]["failed"]:
        reason += " (species %s)" % ", ".join("%s %+.3f < -%.3f" % (s, guards["species"]["per_species"][s]["delta"],
                                                                     guards["species"]["per_species"][s]["threshold"])
                                              for s in guards["species"]["failed"])
    return {
        "format": FORMAT, "protocol": PROTOCOL, "status": AVAILABLE, "v3_applied": True,
        "id": "gate3/%s/%s" % (entry.get("chain"), entry.get("k")), "chain": entry.get("chain"),
        "k": entry.get("k"), "step": entry.get("step"), "tag": entry.get("tag"), "clean": entry.get("clean"),
        "verdict": verdict, "commit_verdict": verdict, "pinned_verdict": pinned_verdict,
        "changed": verdict != pinned_verdict, "reason": reason,
        "p_data": d["p_data"], "p_recipe": d["p_recipe"], "inc": d["inc"], "cand_mean": d["cand_mean"],
        "null_mean": d["null_mean"], "null_sd": d["null_sd"],
        "p_data_le_p_reject": G._le(d["p_data"], pcfg.p_reject),
        "guards": guards, "failed_guards": failed,
        "pinned_failed_guards": [g for g in ("regression", "species", "flips") if not d["guards"][g]["passed"]],
        "species_tolerance": {s: tols[s]["tolerance"] for s in G.SPECIES},
        "species_se": {s: tols[s]["se"] for s in G.SPECIES},
        "attribution": attr,
        "config": {"gate": d["config"], "gate3": dataclasses.asdict(cfg)},
        "inputs": {"ledger_entry_sha256": _sha_text(_canon(entry)), "verified": bool(verify),
                   "scores": entry.get("inputs"),
                   "sidecar": dict(sc_file or {}, weights_sha256=sc.get("weights_sha256"))},
    }


def truth3(entry, sidecar, pinned_cfg, cfg=Gate3Config(), verify=True):
    """The v3 truth verdict of one pinned truth ledger entry: the species
    guard between the with and without arms with tol_s from the sidecar of
    the without arm's first run; P as recorded."""
    if not isinstance(entry, dict) or entry.get("type") != "truth" or "detail" not in entry:
        raise Gate3Error("not a pinned truth ledger entry")
    det = entry["detail"]
    without = (entry.get("inputs") or {}).get("without") or []
    if not without:
        raise Gate3Error("the truth entry lists no 'without' runs")
    if verify:
        rederive_truth(entry, pinned_cfg)
    # The sidecar is bound to the without arm's first run in every mode: its
    # score file (checked against the ledger's sha256) names the weights.
    weights0 = _read_scores(without[:1], "truth without")[0].get("weights_sha256")
    if not weights0:
        raise Gate3Error("the truth entry's first 'without' score names no weights")
    sc, sc_file = load_sidecar(sidecar)
    stamps = det.get("stamps") or {}
    tols = species_tolerances(sc, cfg, weights_sha256=weights0, stamps=stamps,
                              require_production=pinned_cfg.require_production)
    run0 = without[0].get("run_id")
    guard = species_guard(det["species"], tols, a="with", b="without")
    verdict = truth_verdict_of(guard["passed"], det["p"], pinned_cfg)
    return {"format": FORMAT, "protocol": PROTOCOL, "status": AVAILABLE, "v3_applied": True,
            "id": "truth3/%s" % entry.get("k"), "k": entry.get("k"), "step": entry.get("step"),
            "tag": entry.get("tag"), "clean": entry.get("clean"),
            "verdict": verdict, "commit_verdict": verdict, "pinned_verdict": det["verdict"],
            "changed": verdict != det["verdict"], "p": det["p"], "species": guard,
            "species_tolerance": {s: tols[s]["tolerance"] for s in G.SPECIES},
            "tolerance_from": run0,
            "inputs": {"ledger_entry_sha256": _sha_text(_canon(entry)), "verified": bool(verify),
                       "scores": entry.get("inputs"),
                       "sidecar": dict(sc_file or {}, weights_sha256=sc.get("weights_sha256"))}}


UNAVAILABLE_ACCEPT = G.HOLD       # what a pinned ACCEPT commits as when v3 cannot be applied


def unavailable(entry, why, kind="gate"):
    """The record of a step whose v3 decision could not be made (module
    docstring): v3_applied false; commit_verdict is the pinned verdict,
    except that a gate step's pinned ACCEPT commits as HOLD (never an
    accept on a rule that was not applied). The pinned failed guards and
    P_data reading are recorded for the disposition."""
    if kind == "gate":
        d = entry.get("decision") or {}
        pinned = d.get("verdict")
        commit = UNAVAILABLE_ACCEPT if pinned == G.ACCEPT else pinned
        rule = ("pinned ACCEPT held: Protocol v3's species guard could not be applied" if pinned == G.ACCEPT
                else "pinned verdict: v3 cannot make a pinned %s an ACCEPT without its own decision" % pinned)
    else:
        pinned = (entry.get("detail") or {}).get("verdict")
        commit = pinned
        rule = "pinned truth verdict: truth3 could not be applied"
    out = {"format": FORMAT, "protocol": PROTOCOL, "status": UNAVAILABLE, "v3_applied": False,
           "id": ("gate3/%s/%s" % (entry.get("chain"), entry.get("k")) if kind == "gate"
                  else "truth3/%s" % entry.get("k")),
           "chain": entry.get("chain"), "k": entry.get("k"), "step": entry.get("step"), "tag": entry.get("tag"),
           "verdict": None, "commit_verdict": commit, "commit_rule": rule, "pinned_verdict": pinned,
           "changed": False, "reason": why, "inputs": {"ledger_entry_sha256": _sha_text(_canon(entry)),
                                                       "scores": entry.get("inputs")}}
    if kind == "gate":
        try:
            guards = d["guards"]
            out["pinned_failed_guards"] = [g for g in ("regression", "species", "flips") if not guards[g]["passed"]]
            out["failed_guards"] = list(out["pinned_failed_guards"])
            out["p_data"] = d["p_data"]
            out["p_data_le_p_reject"] = G._le(d["p_data"], pinned_config(d).p_reject)
        except (KeyError, TypeError, Gate3Error):
            pass
    return out


# --------------------------------------------------------------- experiment
def sidecar_path(exp_root, run_id, score_path=None):
    """The sidecar beside a run's recorded dev score, else under exp_root."""
    if score_path:
        p = Path(score_path).parent / SIDECAR_NAME
        if p.exists():
            return p
    return Path(exp_root) / "runs" / run_id / "scores" / SIDECAR_NAME


def read_ledger(path):
    entries, raw = [], b""
    with open(path, "rb") as fh:
        raw = fh.read()
    for i, ln in enumerate(raw.decode("utf-8").splitlines(), 1):
        if ln.strip():
            try:
                entries.append(json.loads(ln))
            except ValueError as e:
                raise Gate3Error("%s line %d is not JSON (%s)" % (path, i, e))
    return entries, hashlib.sha256(raw).hexdigest()


def experiment_gate_config(entries):
    """The experiment's pinned GateConfig: its gate_pin/0 entry, else the
    config its first gate decision recorded, else GateConfig()."""
    for e in entries:
        if e.get("type") == "gate_pin":
            return G.GateConfig(**e["config"])
    for e in entries:
        if e.get("type") == "gate":
            return pinned_config(e["decision"])
    return G.GateConfig()


def decide_experiment(exp, cfg=Gate3Config(), verify=True, out=None, write=True, root=None):
    """Re-decide every gate and truth entry of a chain experiment (module
    docstring); returns the document and writes it (atomically) unless
    write=False."""
    root = Path(root) if root else C.INC_DIR / exp
    ledger = root / "ledger.jsonl"
    if not ledger.is_file():
        raise Gate3Error("%s has no ledger" % root)
    entries, ledger_sha = read_ledger(ledger)
    pcfg = experiment_gate_config(entries)
    steps, truth = [], []
    for e in entries:
        if e.get("type") == "gate":
            inc = (e.get("inputs") or {}).get("inc") or {}
            p = sidecar_path(root, inc.get("run_id") or (e.get("incumbent_before") or {}).get("run_id"),
                             inc.get("path"))
            try:
                steps.append(decide(e, p, cfg, verify=verify))
            except (Gate3Error, KeyError, TypeError) as err:
                steps.append(unavailable(e, "%s: %s" % (type(err).__name__, err)))
        elif e.get("type") == "truth":
            wo = ((e.get("inputs") or {}).get("without") or [{}])[0]
            p = sidecar_path(root, wo.get("run_id"), wo.get("path"))
            try:
                truth.append(truth3(e, p, pcfg, cfg, verify=verify))
            except (Gate3Error, KeyError, TypeError) as err:
                truth.append(unavailable(e, "%s: %s" % (type(err).__name__, err), kind="truth"))
    exp_json = root / "exp.json"
    doc = {"format": FORMAT, "protocol": PROTOCOL, "exp": exp, "generated_utc": _utc(),
           "config": {"gate": dataclasses.asdict(pcfg), "gate3": dataclasses.asdict(cfg)},
           "rule": "tol_s = max(%g, %g x SE_s); SE_s = image-bootstrap SE of the incumbent's dev AP for s "
                   "(%d resamples, seed stable_int(%r)); every other guard and threshold is the pinned gate's"
                   % (cfg.species_floor, cfg.species_z, cfg.resamples, cfg.seed_text),
           "inputs": {"ledger": {"path": str(ledger), "sha256": ledger_sha, "entries": len(entries)},
                      "exp_json": {"path": str(exp_json),
                                   "sha256": C.sha256_file(exp_json) if exp_json.is_file() else None},
                      "gate3_module_sha256": C.sha256_file(Path(__file__).resolve()),
                      "gate_module_sha256": C.sha256_file(Path(G.__file__).resolve()), "verified": bool(verify)},
           "steps": steps, "truth": truth,
           "summary": {"steps": len(steps), "v3_applied": sum(1 for s in steps if s["v3_applied"]),
                       "unavailable": [s["id"] for s in steps if not s["v3_applied"]],
                       "unavailable_held": [s["id"] for s in steps if not s["v3_applied"]
                                            and s.get("commit_verdict") != s.get("pinned_verdict")],
                       "truth_unavailable": [t["id"] for t in truth if not t["v3_applied"]],
                       "changed": [s["id"] for s in steps if s.get("changed")],
                       "truth_changed": [t["id"] for t in truth if t.get("changed")]}}
    if write:
        doc["out"] = str(Path(out) if out else root / OUT_NAME)
        _write_json(doc["out"], doc)
    return doc


def _utc():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Protocol v3's per-species species guard over a chain's ledger.")
    ap.add_argument("command", choices=("decide", "show"))
    ap.add_argument("--exp", required=True)
    ap.add_argument("--no-verify", action="store_true",
                    help="decide from the ledger's recorded guards without re-deriving them from the score files")
    ap.add_argument("--out", default=None, help="default INC_DIR/<exp>/gate3.json")
    a = ap.parse_args(argv)
    try:
        doc = decide_experiment(a.exp, verify=not a.no_verify, out=a.out, write=a.command == "decide")
    except Gate3Error as e:
        print("[inc2.gate3] ERROR: %s" % e, file=sys.stderr)
        return 1
    for s in doc["steps"]:
        log("%s %s: %s (pinned %s)%s" % (s["id"], s.get("step"), s.get("verdict"), s["pinned_verdict"],
                                         "" if s["v3_applied"] else " UNAVAILABLE: %s" % s["reason"]))
    for t in doc["truth"]:
        log("%s %s: %s (pinned %s)%s" % (t["id"], t.get("step"), t.get("verdict"), t["pinned_verdict"],
                                         "" if t["v3_applied"] else " UNAVAILABLE: %s" % t["reason"]))
    if a.command == "decide":
        log("wrote %s" % doc["out"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
