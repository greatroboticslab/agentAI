"""The stream-mode lever menu (docs/CONTINUOUS_LOOP.md 6.3): loading, argv
rendering, parameter checks, limits and prices.

The menu is stream_levers.json (levers L15-L28, the network probe LP, the
sub-levers of L16 and L23, the hold release LH; cards X13-X17; the limits of
6.6; the gated R2 levers and the envelope levers of 6.5). The thresholds are
stream_thresholds.json. A campaign's domain facts come from its stream-domain
config (stream_domains/<domain>.json), never from this module: the code names
no class, source, exam or lab (contract S13).

A lever's argv is rendered from its template (levers.json's grammar, plus
'{pkg}', the campaign's protocol package); executor.ARGV_FORMS re-renders the
same command from the policy parameters, token for token, and the executor
runs only what the policy table checked (tests/test_stream_ap_units.py pins
the equality for every lever).

Prices follow the cost model of contract 5.6 with the rates of the domain
config (`cost`): a price is an upper-leaning estimate in V100 GPU-hours, the
unit the policy rows' est_su formula charges. Nothing here reads an exam.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path

from . import model as M

PKG_DIR = Path(__file__).resolve().parent
MENU_FILE = PKG_DIR / "stream_levers.json"
THRESHOLDS_FILE = PKG_DIR / "stream_thresholds.json"
DOMAINS_DIR = PKG_DIR / "stream_domains"
TOOLS_DIR = PKG_DIR.parent
CODE_ROOT = PKG_DIR.parents[2]                 # the directory holding weed_optimizer_framework/
DOMAIN_FORMAT = "inc-autopilot/stream-domain/1"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_CACHE = {}


class LeverError(ValueError):
    """A lever that cannot be rendered, checked or priced as asked."""


def _read_json(path):
    p = Path(path)
    key = ("json", str(p), p.stat().st_mtime_ns, p.stat().st_size)
    if key not in _CACHE:
        with open(str(p), "r", encoding="utf-8") as fh:
            _CACHE[key] = json.load(fh)
    return copy.deepcopy(_CACHE[key])


def sha256_file(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ the menu
def load_menu(path=None):
    return _read_json(path or MENU_FILE)


def row(lid, menu=None):
    menu = menu or load_menu()
    r = (menu.get("levers") or {}).get(lid)
    if not isinstance(r, dict):
        raise LeverError("%r is not on the stream lever menu" % (lid,))
    return dict(r, id=lid)


def lever_ids(menu=None):
    return sorted((menu or load_menu()).get("levers") or {})


def family(lid, menu=None):
    """The lever a sub-lever belongs to (L16I -> L16, L23B -> L23); itself otherwise.
    Limits and the gated/envelope lists are keyed on it."""
    r = (menu or load_menu()).get("levers", {}).get(lid) or {}
    return str(r.get("lever") or lid)


def actions_of(lid, menu=None):
    r = row(lid, menu)
    return tuple(r.get("actions") or (r["policy_action"],))


def stream_actions(menu=None):
    """{policy action: [lever ids]} of every stream lever."""
    out = {}
    for lid, r in sorted(((menu or load_menu()).get("levers") or {}).items()):
        out.setdefault(r.get("policy_action"), []).append(lid)
    return out


def limits(lid, menu=None):
    menu = menu or load_menu()
    return dict((menu.get("limits") or {}).get(family(lid, menu)) or {})


def gated_r2(menu=None):
    return tuple(((menu or load_menu()).get("gated_r2") or {}).get("value") or ())


def envelope_levers(menu=None):
    return tuple(((menu or load_menu()).get("envelope") or {}).get("value") or ())


def packages(menu=None):
    return tuple(((menu or load_menu()).get("packages") or {}).get("value") or ())


def card(cid, menu=None):
    menu = menu or load_menu()
    return (menu.get("cards") or {}).get(cid)


# ------------------------------------------------------------- thresholds
class Missing(KeyError):
    """A threshold stream_thresholds.json does not declare."""


def load_thresholds(path=None):
    return _read_json(path or THRESHOLDS_FILE)


def t(th, block, key):
    try:
        return th[block][key]["value"]
    except (KeyError, TypeError):
        raise Missing("%s.%s" % (block, key))


# ------------------------------------------------------------ domain config
def domain_path(domain_or_path):
    s = str(domain_or_path or "")
    if s.endswith(".json") or "/" in s:
        p = Path(s)
        if not p.is_absolute():
            for base in (TOOLS_DIR, CODE_ROOT, PKG_DIR):
                if (base / s).is_file():
                    return base / s
        return p
    return DOMAINS_DIR / ("%s.json" % s)


def load_domain(domain_or_path):
    """The stream-domain config, with `_path` and `_sha256` added. Raises LeverError."""
    p = domain_path(domain_or_path)
    try:
        d = _read_json(p)
    except (OSError, ValueError) as e:
        raise LeverError("the stream-domain config %s cannot be read (%s)" % (p, type(e).__name__))
    if not isinstance(d, dict) or d.get("format") != DOMAIN_FORMAT:
        raise LeverError("%s is not a stream-domain config (%s)" % (p, DOMAIN_FORMAT))
    for k in ("domain", "sid", "protocol_package", "funnel_domain", "increment"):
        if k not in d:
            raise LeverError("the stream-domain config %s has no %r" % (p, k))
    d["_path"] = str(p)
    d["_sha256"] = sha256_file(p)
    return d


def funnel_domain(dom):
    """The funnel domain config the stream domain points at (targets, exams),
    resolved like model.exam_splits does (a name or a path)."""
    from ..funnel import domain as FD
    ref = dom.get("funnel_domain")
    p = Path(str(ref))
    if str(ref).endswith(".json") and not p.is_absolute():
        for base in (TOOLS_DIR, CODE_ROOT):
            if (base / p).is_file():
                ref = str(base / p)
                break
    return FD.load(ref)


def funnel_ref(dom):
    """What model.exam_splits takes for this domain: a name or an absolute path."""
    ref = dom.get("funnel_domain")
    p = Path(str(ref))
    if str(ref).endswith(".json") and not p.is_absolute():
        for base in (TOOLS_DIR, CODE_ROOT):
            if (base / p).is_file():
                return str(base / p)
    return ref


def target_names(dom):
    """The target class names in id order (from the funnel domain config)."""
    fd = funnel_domain(dom)
    rows = ((fd.raw or {}).get("classes") or {}).get("targets") or []
    return [r.get("name") for r in sorted(rows, key=lambda r: int(r.get("id", 0)))]


def cost(dom, key):
    try:
        v = dom["cost"][key]
    except (KeyError, TypeError):
        raise LeverError("the stream-domain config has no cost.%s" % key)
    return v


def inc_path(rel, inc_dir=None):
    """An INC_DIR-relative path on the cluster (model.CLUSTER_INC_DIR)."""
    return "%s/%s" % (str(inc_dir or M.CLUSTER_INC_DIR).rstrip("/"), str(rel).lstrip("/"))


def never_train_slugs(dom):
    """(set, "") of the never-train slugs the stream-domain config names, read
    by parsing the trainer's source (ast.literal_eval of the assignment), never
    by importing it; (None, why) when it cannot be read (fail closed: the
    caller collects nothing)."""
    import ast
    spec = dom.get("never_train") or {}
    rel, attr = spec.get("source"), spec.get("attr")
    if not rel or not attr:
        return None, "the stream-domain config names no never-train source"
    p = Path(str(rel))
    if not p.is_absolute():
        p = CODE_ROOT / p
    try:
        tree = ast.parse(p.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError) as e:
        return None, "%s unreadable (%s)" % (rel, type(e).__name__)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(x, "id", None) == attr for x in node.targets):
            try:
                return set(str(v) for v in ast.literal_eval(node.value)), ""
            except (ValueError, TypeError) as e:
                return None, "%s.%s is not a literal (%s)" % (rel, attr, type(e).__name__)
    return None, "%s holds no %s assignment" % (rel, attr)


# -------------------------------------------------------------- rendering
def _fill(tok, params):
    def sub(m):
        k = m.group(1)
        v = params.get(k)
        if v is None:
            raise LeverError("the argv needs %r" % k)
        return str(v)
    return re.sub(r"\{([a-z_]+)\}", sub, tok)


def policy_params(lid, params, menu=None):
    """The policy parameters of lever lid: its fixed values (e.g. verb) over the
    given params. The protocol package stays a parameter ('pkg')."""
    r = row(lid, menu)
    out = dict(params or {})
    for k, v in (r.get("fixed") or {}).items():
        if k in out and str(out[k]) != str(v):
            raise LeverError("lever %s fixes %s = %r; got %r" % (lid, k, v, out[k]))
        out[k] = v
    return out


def render(lid, params, menu=None):
    """The exact command of lever lid with these params (tokens); None for a
    lever without a command (the sync hook). Raises LeverError."""
    r = row(lid, menu)
    tmpl = r.get("argv")
    if tmpl is None:
        return None
    p = policy_params(lid, params, menu)
    if any("{pkg}" in str(x) for x in tmpl if isinstance(x, str)):
        pkgs = packages(menu)
        if p.get("pkg") not in pkgs:
            raise LeverError("protocol package %r is not one of %s" % (p.get("pkg"), list(pkgs)))
    out = []
    for tok in tmpl:
        if isinstance(tok, dict):
            if p.get(tok.get("if")) is None:
                continue
            out += [_fill(x, p) for x in tok.get("tokens") or []]
        else:
            out.append(_fill(tok, p))
    return out


def check_params(lid, params, menu=None):
    """(ok, reasons) of the policy parameters against the lever's policy row
    (brain/policy_actions.json bounds; policy._check_params) and its fixed values."""
    from ..brain import policy as POL
    r = row(lid, menu)
    desc = POL.describe(r["policy_action"])
    if not desc.get("known"):
        return False, ["no valid policy row for %s (%s)" % (r["policy_action"], desc.get("reason"))]
    try:
        p = policy_params(lid, params, menu)
    except LeverError as e:
        return False, [str(e)]
    return POL._check_params(p, desc["param_bounds"])


# ------------------------------------------------------------------- prices
def _hours(images, epochs, ms):
    return float(images) * float(epochs) * float(ms) / 3.6e6


def arm_factor(dom, arm=None):
    """GPU time of a capacity arm (an inc2.recipes.ARMS id) relative to the
    default arm, from the stream-domain config (est.)."""
    if arm in (None, ""):
        return 1.0
    a = ((dom.get("capacity") or {}).get("arms") or {}).get(str(arm))
    if not isinstance(a, dict):
        raise LeverError("no capacity arm %r in the stream-domain config" % (arm,))
    return float(a.get("cost_factor") or 1.0)


def step_hours(dom, pool_images, m, recipes=("r0",), factor=1.0):
    """(chain hours per step over the recipes, truth hours per step)."""
    seeds = int(cost(dom, "seeds"))
    chain = 0.0
    for r in recipes:
        ep = (cost(dom, "chain_epochs") or {}).get(r)
        if ep is None:
            raise LeverError("recipe %r has no epochs in the stream-domain config" % r)
        chain += _hours(seeds * (pool_images + m) + seeds * pool_images, ep, cost(dom, "incr_ms_per_image_epoch"))
    truth = _hours(seeds * (pool_images + m), cost(dom, "cold_epochs"), cost(dom, "cold_ms_per_image_epoch"))
    return chain * factor, truth * factor


def truth_every(dom, th, pool_images, m, recipes=("r0",), factor=1.0):
    """L-4: 1 (truth on every step) unless one step with its truth arm costs
    more than capacity.truth_step_hours_max; then ceil(cost / max)."""
    chain, truth = step_hours(dom, pool_images, m, recipes, factor)
    cap = float(t(th, "capacity", "truth_step_hours_max"))
    per = chain + truth
    return 1 if per <= cap else int(math.ceil(per / cap))


def price(lid, params, dom, info=None):
    """(est_gpu_hours, detail) of lever lid. `info`: {pool_images, images,
    n_suspect, recipes} as the ticker knows them. Raises LeverError."""
    r = row(lid)
    kind = r.get("estimator")
    info = dict(info or {})
    p = dict(params or {})
    if info.get("M"):
        p.setdefault("m", int(info["M"]))
    build = float(cost(dom, "build_job_hours"))
    n = int(info.get("pool_images") or (dom.get("increment") or {}).get("base_images") or 0)
    m = int(p.get("m") or (dom.get("increment") or {}).get("M") or 0)
    seeds = int(cost(dom, "seeds"))
    cold_ms, ep_cold = cost(dom, "cold_ms_per_image_epoch"), cost(dom, "cold_epochs")
    detail = {"estimator": kind, "rates": "stream-domain cost (contract 5.6, est.)"}
    if kind in ("zero",):
        return 0.0, dict(detail, why="no allocation")
    if kind == "zero_job":
        return round(build, 3), dict(detail, build_job_hours=build)
    if kind == "probe":
        h = float(cost(dom, "probe_hours"))
        return h, dict(detail, job_hours=h)
    if kind == "collect":
        h = float(cost(dom, "collect_job_hours"))
        return h, dict(detail, job_hours=h, why="the collect job's walltime, an upper bound")
    if kind == "intake":
        h = float(cost(dom, "intake_job_hours"))
        return h, dict(detail, job_hours=h)
    if kind == "admit":
        imgs = int(info.get("images") or 0)
        h = float(cost(dom, "admit_fixed_hours")) + imgs * float(cost(dom, "admit_seconds_per_1000_images")) / 1000.0 / 3600.0
        if not imgs:
            h += 1.0                       # an unknown batch size: the 50,000-image cap's hour on top
        return round(h, 3), dict(detail, images=imgs)
    if kind == "splits":
        h = float(cost(dom, "splits_hours"))
        return h, dict(detail, job_hours=h)
    factor = arm_factor(dom, p.get("arm") or info.get("arm"))
    recipes = tuple(x for x in str(p.get("recipes") or info.get("recipes") or "r0").split(",") if x)
    finals = float(cost(dom, "finals_hours"))
    if kind == "segment":
        k = int(p.get("k") or 1)
        every = int(info.get("truth_every") or 1)
        base = _hours(seeds * n, ep_cold, cold_ms) * factor
        chain, truth = step_hours(dom, n, m, recipes, factor)
        n_truth = int(math.ceil(k / float(every)))
        exp_h = base + k * chain + n_truth * truth + finals
        return round(exp_h + build, 3), dict(detail, base=round(base, 3), chain_per_step=round(chain, 3),
                                             truth_per_step=round(truth, 3), truth_steps=n_truth, k=k,
                                             pool_images=n, m=m, recipes=list(recipes), factor=factor,
                                             finals=finals, build_job_hours=build)
    if kind == "stage_c":
        base = _hours(seeds * max(n - m, 0), ep_cold, cold_ms)
        chain, truth = step_hours(dom, max(n - m, 0), m, recipes, 1.0)
        exp_h = base + chain + truth + finals
        return round(exp_h + build, 3), dict(detail, base=round(base, 3), chain=round(chain, 3),
                                             truth=round(truth, 3), finals=finals, build_job_hours=build)
    if kind == "milestone":
        ms = int(cost(dom, "milestone_seeds"))
        exp_h = _hours(ms * n, ep_cold, cold_ms) * factor + finals
        return round(exp_h + build, 3), dict(detail, seeds=ms, pool_images=n, factor=factor, build_job_hours=build)
    if kind == "baseline":
        imgs = int(info.get("images") or n)
        s = len([x for x in str(p.get("seeds") or "").split(",") if x != ""]) or seeds
        exp_h = _hours(s * imgs, ep_cold, cold_ms) * factor + finals
        return round(exp_h + build, 3), dict(detail, seeds=s, images=imgs, factor=factor, build_job_hours=build)
    if kind == "pilot4":
        h = float((dom.get("stage_a") or {}).get("est_gpu_hours") or 18.0)
        return round(h + build, 3), dict(detail, experiment_hours=h, build_job_hours=build,
                                         why="contract 5.1: about 17-18 GPU-h (est.)")
    if kind == "bisect":
        k = max(1, int(info.get("n_suspect") or 4))
        each = _hours(seeds * (n + m), ep_cold, cold_ms) * factor
        return round(k * each + finals + build, 3), dict(detail, arms=k, each=round(each, 3), build_job_hours=build)
    raise LeverError("lever %s has no estimator %r" % (lid, kind))


# ---------------------------------------------------------------- proposals
def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
                          .encode("utf-8")).hexdigest()


def proposal(campaign, lid, params, trigger=(), cites=(), lane=None, parent_exp=None, child_exp=None,
             est=None, estimate=None, attempt=0, extra=None, menu=None):
    """A model.proposal-shaped dict for lever lid, with a deterministic id (the
    same lever, command, params and attempt give the same id, so a proposal is
    filed and run once)."""
    r = row(lid, menu)
    pp = policy_params(lid, params, menu)
    argv = render(lid, pp, menu)
    if est is not None:
        pp["est_gpu_hours"] = float(est)
    pid = _sha([campaign, lid, r["policy_action"], argv, pp, parent_exp, child_exp, int(attempt)])[:32]
    out = {"id": pid, "lever": lid, "family": family(lid, menu), "argv": argv, "params": pp,
           "policy_action": r["policy_action"], "risk": r["risk"], "trigger": list(trigger),
           "cites": list(cites), "lit": [], "control": r.get("control", ""), "success": r.get("success", ""),
           "falsifier": r.get("falsifier", ""), "est_gpu_hours": float(est or 0.0),
           "proposed_by": M.AUTOPILOT_ACTOR, "lane": lane or r.get("lane"), "follow": r.get("follow"),
           "parent_exp": parent_exp, "child_exp": child_exp, "estimate": estimate, "attempt": int(attempt)}
    if extra:
        out.update(extra)
    return out
