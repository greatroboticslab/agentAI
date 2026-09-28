"""The funnel audit's command line (docs/FUNNEL_AUDIT_RUNNER.md 5.6.1).

    python -m <package>.tools.funnel <verb> --prereg PATH [--out DIR] [--force] [--quiet] [--testing] [options]
    python -m <package>.tools.funnel sbatch-args VERB

Every verb but sbatch-args reads the pre-registration (--prereg; the domain is
the one it names, through domain.load_pair, so there is no --domain) and calls
one engine function, imported when the verb runs, so a verb never needs the
modules of another. Its outputs go to --out (default FUNNEL_DIR, and R1_DIR
for recover). One closing line "[funnel] <verb>: <summary>" goes to stdout.

Exit codes: 0 done; 2 refused (any FunnelError, the message on stderr; an
argument error is a CLIError); 1 crash (any other exception, with its
traceback).

Verbs, their options and the function each one calls:
  census        --taxonomy PATH --known-items PATH      adapter.census
                --summaries-only [--census-v0 PATH]     adapter.ledger_from_summaries
  leak                                                  leak.run
  embed-judges  --stage embed|judges|all                embed.embed_crops (every shard, or --shard of
                --shard i --nshards n                   --nshards), then judges.score_all
                --refetch                               embed.embed_table on the refetch crop table
  qualify       [--rl]                                  qualify.judges / qualify.rl
  draw                                                  draw.draw
  sheets                                                sheets.run
  rl-b          --endpoint URL --model TAG              rl.OllamaClient(...).check_vision(), rl.run_rl_b
                --sheet-dirs D ... --max-attempts 3
  ingest        --answers DIR ...                       rl.ingest
  estimate                                              estimate.evaluate, estimate.render_md
  map           --part geometry|relation                relation.run_geometry / relation.run_relation
  recover       --audit PATH --maps PATH --policy L     recover.run
                --arms --realloop EXP --base PATH       recover.arms
  fetch         --what LIST --names-from PATH           fetch.* (lab only), then fetch.write_manifest
                --sources PATH ... --slug SLUG ...
  sbatch-args   VERB                                    prints SBATCH_RESOURCES[VERB_CLASS[VERB]]

Arguments are passed as the runner pins them: a parameter named prereg gets
the loaded Prereg, prereg_path its path, domain the loaded Domain, adapter the
adapter module the config names (adapters.load). --force, --quiet and
--testing are passed to a function that has a parameter of that name; --force
given to one that has none is refused, never dropped. A function with a
prereg or domain keyword that the call does not bind gets the loaded objects
too, so every output header records the --prereg given here.

GPU verbs (VERB_CLASS gpu and gpu_large) refuse to run without a visible CUDA
device unless --testing, which the tests use with fake models. fetch refuses
inside a Slurm job: compute nodes have no network, and the lab runs it.

Choices this file makes where the runner leaves room are listed under "G-loop"
at the end of docs/FUNNEL_AUDIT_RUNNER.md.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

from . import CLIError, FUNNEL_DIR, FunnelError, R1_DIR, STEP1_DIR, json_text, strip_volatile, write_json_atomic
from ..inc import common as C

VERBS = ("census", "leak", "embed-judges", "qualify", "draw", "sheets", "rl-b", "ingest", "estimate", "map",
         "recover", "fetch", "sbatch-args")
VERB_CLASS = {"census": "cpu", "leak": "gpu", "embed-judges": "gpu", "qualify": "cpu", "draw": "cpu",
              "sheets": "cpu", "rl-b": "gpu_large", "ingest": "cpu", "estimate": "cpu", "map": "cpu",
              "recover": "cpu", "fetch": "lab"}
SBATCH_RESOURCES = {                                    # extra sbatch flags, before the script name
    "cpu": [],                                          # run_inc_funnel.sh's own #SBATCH lines (one V100, idle)
    "gpu": [],                                          # the same lines; the verb uses the V100
    "gpu_large": ["--gres=gpu:h100-80:1", "--cpus-per-task=12", "--mem=80G"]}
GPU_CLASSES = ("gpu", "gpu_large")
JOB_SCRIPT = "run_inc_funnel.sh"

EMBED_STAGES = ("embed", "judges", "all")
MAP_PARTS = ("geometry", "relation")
FETCH_WHAT = ("taxonomy", "known-items", "cards", "kt7", "refetch")      # run in this order
RECOVER_POLICIES = ("R-A", "R-C", "R-T", "R-V", "R-J", "R-F")
ENDPOINT_ENV = "FUNNEL_OLLAMA_ENDPOINT"
RL_B = "RL-B"
MAX_ATTEMPTS = 3

CENSUS_V0 = "census_v0.json"
LEDGER_NAME = "funnel_ledger.json"
TAXONOMY_NAME = "taxonomy_cache.json"
KNOWN_ITEMS_NAME = "known_items_v1.json"
AUDIT_NAME = "audit_v1.json"
AUDIT_MD = "audit_v1.md"
MAPS_NAME = "class_maps.json"
SHEET_DIRS = ("sheets_v1", "sheets_v1_cluster")
ANSWERS_DIR = "rl_answers"
EMB_DIR = "emb_dinov2"
REFETCH_TABLE = Path("refetch") / "crops_refetch.csv"
EMBEDDER_SUFFIX = "_embedder"


def prog():
    return "python -m %s" % __package__


def log(msg, quiet=False):
    if not quiet:
        print("[funnel] %s" % msg, flush=True)


# ------------------------------------------------------------------ helpers
def _module(name):
    """The engine module `name` of this package, imported now. A module that
    does not exist yet is a refusal; any other import failure is a crash."""
    full = "%s.%s" % (__package__, name)
    try:
        return importlib.import_module(full)
    except ModuleNotFoundError as e:
        if e.name == full:
            raise CLIError("the engine module %s is not installed in this copy of the package" % full)
        raise


def _call(fn, *args, optional=None, context=None, **kwargs):
    """fn(*args, **kwargs), plus:
      * every optional flag (force, quiet, testing) that is set and that fn
        accepts; a set force that fn does not accept is refused;
      * every context object (the loaded prereg and domain) that fn has a
        parameter of that name for and that the call does not bind already,
        so an engine function records the CLI's --prereg, never one it would
        look up by itself."""
    try:
        sig = inspect.signature(fn)
        params = sig.parameters
    except (TypeError, ValueError):
        sig, params = None, {}
    var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    for k, v in sorted((optional or {}).items()):
        if not v:
            continue
        if k in params or var_kw:
            kwargs[k] = v
        elif k == "force":
            raise CLIError("--force: %s.%s has no force parameter" % (getattr(fn, "__module__", "?"),
                                                                     getattr(fn, "__name__", repr(fn))))
    if context and sig is not None:
        try:
            bound = sig.bind_partial(*args, **kwargs).arguments
        except TypeError:
            bound = {}
        for k, v in sorted(context.items()):
            if k in params and k not in bound and params[k].kind in (params[k].POSITIONAL_OR_KEYWORD,
                                                                     params[k].KEYWORD_ONLY):
                kwargs[k] = v
    return fn(*args, **kwargs)


def _abs(path):
    return Path(os.path.abspath(str(path)))


def _file(path, what):
    p = _abs(path)
    if not p.is_file():
        raise CLIError("%s %s does not exist" % (what, p))
    return p


def _write_bytes_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_text_atomic(path, text):
    _write_bytes_atomic(path, text.encode("utf-8"))


def _read_bytes(path):
    try:
        return Path(path).read_bytes()
    except OSError:
        return None


def _same_record(a, b):
    """Two JSON texts (bytes) that hold the same record once the volatile keys
    (built_utc, seconds, hostname, slurm_job_id) are removed."""
    try:
        return strip_volatile(json.loads(a)) == strip_volatile(json.loads(b))
    except (TypeError, ValueError):
        return False


def cuda_available():
    """A CUDA device is visible: torch.cuda when torch imports, else nvidia-smi -L."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        try:
            return bool(torch.cuda.is_available())
        except Exception:                       # noqa: BLE001 - a broken CUDA stack is no device
            return False
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        r = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0 and "GPU" in r.stdout


def sbatch_args(verb):
    """The extra sbatch flags of a verb, from SBATCH_RESOURCES and VERB_CLASS."""
    if verb not in VERB_CLASS:
        raise CLIError("sbatch-args: %r is not a verb of %s" % (verb, [v for v in VERBS if v in VERB_CLASS]))
    cls = VERB_CLASS[verb]
    if cls not in SBATCH_RESOURCES:
        raise CLIError("sbatch-args: %s runs on the lab (class %s), never as a cluster job" % (verb, cls))
    return list(SBATCH_RESOURCES[cls])


def step1_embedder(adapter):
    """(tag, factory) of the adapter's own image embedder: the one function of
    the adapter interface (adapters.INTERFACE) whose name ends in
    EMBEDDER_SUFFIX. Its features of a crop table are emb_<tag>_<table>.npz."""
    names = [n for n in _module("adapters").INTERFACE if n.endswith(EMBEDDER_SUFFIX)]
    names = [n for n in names if callable(getattr(adapter, n, None))]
    if len(names) != 1:
        raise CLIError("the adapter must offer exactly one *%s function, found %s" % (EMBEDDER_SUFFIX, names))
    return names[0][:-len(EMBEDDER_SUFFIX)], getattr(adapter, names[0])


def realloop_dir(value):
    """--realloop: a path (holding a separator), or an experiment name under INC_DIR."""
    s = str(value)
    d = _abs(s) if ("/" in s or os.sep in s) else Path(C.INC_DIR) / s
    if not (d / "exp.json").is_file():
        raise CLIError("--realloop %s: no experiment at %s (no exp.json)" % (value, d))
    return d


def _summary(result, out):
    parts = []
    if isinstance(result, dict):
        for k in ("format", "status", "valid", "locked"):
            if isinstance(result.get(k), (str, bool, int, float)):
                parts.append("%s=%s" % (k, result[k]))
        scalars = [(k, v) for k, v in sorted(result.items())
                   if k not in ("format", "status", "valid", "locked", "built_utc", "testing")
                   and isinstance(v, (bool, int, float, str)) and len(str(v)) <= 64]
        parts += ["%s=%s" % kv for kv in scalars[:6]]
        nested = [k for k, v in sorted(result.items()) if isinstance(v, (dict, list))]
        if nested:
            parts.append("sections=%s" % ",".join(nested[:8]) + (",..." if len(nested) > 8 else ""))
    elif isinstance(result, (list, tuple)):
        parts.append("%d item(s)" % len(result))
    elif result is None:
        parts.append("done")
    else:
        parts.append(str(result)[:120])
    parts.append("out=%s" % out)
    return "; ".join(parts)


class Ctx(object):
    """What every verb handler gets: the parsed arguments, the prereg and its
    domain, the output directory and the adapter (loaded on first use)."""

    def __init__(self, args, prereg_path, prereg, domain, out):
        self.args = args
        self.prereg_path = prereg_path
        self.prereg = prereg
        self.domain = domain
        self.out = out
        self._adapter = None

    def adapter(self):
        if self._adapter is None:
            self._adapter = _module("adapters").load(self.domain.adapter)
        return self._adapter

    @property
    def optional(self):
        a = self.args
        return {"force": a.force, "quiet": a.quiet, "testing": a.testing}

    def call(self, fn, *args, extra_context=None, **kwargs):
        """_call with this run's flags, and its prereg and domain (plus
        extra_context) as context."""
        context = dict({"prereg": self.prereg, "domain": self.domain}, **(extra_context or {}))
        return _call(fn, *args, optional=self.optional, context=context, **kwargs)


# ------------------------------------------------------------- the verbs
def do_census(ctx):
    a, ad = ctx.args, ctx.adapter()
    if a.summaries_only:
        if a.taxonomy or a.known_items:
            raise CLIError("census --summaries-only reads the Step 1 summaries and census_v0 only; "
                           "--taxonomy and --known-items belong to the full census")
        v0 = _file(a.census_v0 or (ctx.out / CENSUS_V0), "census_v0")
        return ctx.call(ad.ledger_from_summaries, ctx.domain, STEP1_DIR, v0, ctx.out / LEDGER_NAME)
    if a.census_v0:
        raise CLIError("--census-v0 applies to census --summaries-only")
    tax = _abs(a.taxonomy or (ctx.out / TAXONOMY_NAME))
    if not tax.is_file():
        raise CLIError("no resolution for any name: %s does not exist: run fetch --what taxonomy (lever L12)"
                       % tax)
    if a.known_items:
        known = _file(a.known_items, "--known-items")
    else:
        known = ctx.out / KNOWN_ITEMS_NAME
        known = known if known.is_file() else None
    return ctx.call(ad.census, ctx.prereg, ctx.domain, ctx.out, tax, known_items=known)


def do_leak(ctx):
    return ctx.call(_module("leak").run, ctx.prereg, ctx.domain, ctx.out, ctx.adapter())


def do_embed_judges(ctx):
    a = ctx.args
    E = _module("embed")
    if a.refetch:
        if a.stage != "embed" or a.shard is not None or a.nshards is not None:
            raise CLIError("--refetch embeds the refetch crop table in one job: --stage embed, no --shard or "
                           "--nshards")
        table = _file(ctx.out / REFETCH_TABLE, "the refetch crop table")
        model, pooling = E.features_config(ctx.domain)
        tag, factory = step1_embedder(ctx.adapter())
        return {"refetch_table": str(table),
                "dinov2": ctx.call(E.embed_table, table, ctx.out / "emb_dinov2_refetch.npz",
                                   E.Dinov2Embedder(model, pooling)),
                tag: ctx.call(E.embed_table, table, ctx.out / ("emb_%s_refetch.npz" % tag), factory())}
    if a.stage == "judges" and (a.shard is not None or a.nshards is not None):
        raise CLIError("--shard and --nshards belong to --stage embed or all")
    if a.shard is not None and a.nshards is None:
        raise CLIError("--shard needs --nshards")
    out = {}
    if a.stage in ("embed", "all"):
        n = 1 if a.nshards is None else a.nshards
        shards = [a.shard] if a.shard is not None else list(range(n))
        crops = ctx.adapter().crop_table()
        model, pooling = E.features_config(ctx.domain)
        out["shards"] = {}
        for s in shards:
            meta = ctx.call(E.embed_crops, crops, ctx.out / EMB_DIR, s, n, model=model, pooling=pooling)
            out["shards"]["%d_of_%d" % (s, n)] = {k: meta.get(k) for k in ("crops", "embedder", "crops_sha256")} \
                if isinstance(meta, dict) else meta
    if a.stage in ("judges", "all"):
        out["judges"] = ctx.call(_module("judges").score_all, ctx.prereg, ctx.domain, ctx.out, ctx.adapter())
    return out


def do_qualify(ctx):
    Q = _module("qualify")
    if ctx.args.rl:
        return ctx.call(Q.rl, ctx.prereg, ctx.domain, ctx.out)
    return ctx.call(Q.judges, ctx.prereg, ctx.domain, ctx.out, ctx.adapter())


def do_draw(ctx):
    return ctx.call(_module("draw").draw, ctx.prereg_path, ctx.out, ctx.adapter())


def do_sheets(ctx):
    return ctx.call(_module("sheets").run, ctx.prereg, ctx.domain, ctx.out, ctx.adapter())


def do_rl_b(ctx):
    a = ctx.args
    endpoint = a.endpoint or os.environ.get(ENDPOINT_ENV)
    if not endpoint:
        raise CLIError("rl-b needs --endpoint or $%s (run_inc_funnel.sh starts ollama and sets it)" % ENDPOINT_ENV)
    backends = ((ctx.domain.raw.get("reference_labeller") or {}).get("backends") or {})
    want = (backends.get(RL_B) or {}).get("model")
    if not want:
        raise CLIError("the domain config names no reference_labeller.backends.%s.model" % RL_B)
    model = a.model or want
    if model != want:
        raise CLIError("--model %s is not the configured %s model %s" % (model, RL_B, want))
    if a.max_attempts < 1:
        raise CLIError("--max-attempts must be >= 1, got %d" % a.max_attempts)
    if a.sheet_dirs:
        dirs = [_abs(d) for d in a.sheet_dirs]
    else:
        dirs = [ctx.out / d for d in SHEET_DIRS if (ctx.out / d).is_dir()]
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing or not dirs:
        raise CLIError("rl-b: no sheet directory to answer (missing: %s; run sheets first)"
                       % (missing or [str(ctx.out / d) for d in SHEET_DIRS]))
    R = _module("rl")
    client = R.OllamaClient(endpoint, model)
    client.check_vision()
    return ctx.call(R.run_rl_b, ctx.prereg, ctx.domain, ctx.out, client, dirs, max_attempts=a.max_attempts)


def do_ingest(ctx):
    a = ctx.args
    if a.answers:
        dirs = [_abs(d) for d in a.answers]
    else:
        root = ctx.out / ANSWERS_DIR
        dirs = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    missing = [str(d) for d in dirs if not d.is_dir()]
    if missing or not dirs:
        raise CLIError("ingest: no answer directory (missing: %s; answers live in %s/<backend>/)"
                       % (missing or "none given", ctx.out / ANSWERS_DIR))
    return ctx.call(_module("rl").ingest, ctx.prereg, ctx.domain, ctx.out, dirs)


def do_estimate(ctx):
    """estimate.evaluate, which writes audit_v1.json and audit_v1.md itself
    (with a fresh built_utc) and raises EstimateError after writing an invalid
    audit. A rerun that reproduces the audit on disk (volatile keys aside)
    leaves both files' bytes as they were, so nothing that recorded the
    audit's sha256 (recover, the realloop build) goes stale; an audit the
    evaluator returned without writing is written here."""
    ES = _module("estimate")
    path, md = ctx.out / AUDIT_NAME, ctx.out / AUDIT_MD
    before, before_md = _read_bytes(path), _read_bytes(md)
    try:
        audit = ctx.call(ES.evaluate, ctx.prereg_path, ctx.out, ctx.adapter())
        if not isinstance(audit, dict):
            raise CLIError("estimate.evaluate returned %s, not the audit" % type(audit).__name__)
        on_disk = _read_bytes(path)
        if on_disk is None or not _same_record(on_disk, json_text(audit).encode("utf-8")):
            write_json_atomic(path, audit)
            _write_text_atomic(md, ES.render_md(audit))
        elif _read_bytes(md) is None:
            _write_text_atomic(md, ES.render_md(audit))
    finally:
        after = _read_bytes(path)
        if before is not None and after is not None and after != before and _same_record(before, after):
            _write_bytes_atomic(path, before)
            if before_md is not None:
                _write_bytes_atomic(md, before_md)
    return audit


def do_map(ctx):
    R = _module("relation")
    fn = R.run_geometry if ctx.args.part == "geometry" else R.run_relation
    return ctx.call(fn, ctx.prereg, ctx.domain, ctx.out, ctx.adapter())


def do_recover(ctx):
    a = ctx.args
    RC = _module("recover")
    if a.arms:
        if a.audit or a.maps or a.policy:
            raise CLIError("recover --arms builds the realloop_v2 arms; --audit, --maps and --policy belong to the "
                           "recovery itself")
        if not a.realloop or not a.base:
            raise CLIError("recover --arms needs --realloop EXP and --base PATH")
        return ctx.call(RC.arms, realloop_dir(a.realloop), ctx.out, _file(a.base, "--base"), ctx.adapter())
    if a.realloop or a.base:
        raise CLIError("--realloop and --base belong to recover --arms")
    if not (a.audit and a.maps and a.policy):
        raise CLIError("recover needs --audit, --maps and --policy")
    audit, maps = _file(a.audit, "--audit"), _file(a.maps, "--maps")
    if audit.name != AUDIT_NAME or maps.name != MAPS_NAME or audit.parent != maps.parent:
        raise CLIError("recover reads %s and %s from one funnel directory; got --audit %s and --maps %s"
                       % (AUDIT_NAME, MAPS_NAME, audit, maps))
    policies = [p.strip() for p in a.policy.split(",") if p.strip()]
    bad = [p for p in policies if p not in RECOVER_POLICIES]
    if not policies or bad or len(set(policies)) != len(policies):
        raise CLIError("--policy must list distinct policies among %s, got %r" % (list(RECOVER_POLICIES), a.policy))
    return ctx.call(RC.run, ctx.prereg, ctx.domain, audit.parent, ctx.out, policies, ctx.adapter(),
                    extra_context={"audit_path": audit, "maps_path": maps})


def do_fetch(ctx):
    a = ctx.args
    if os.environ.get("SLURM_JOB_ID"):
        raise CLIError("fetch runs on the lab (network); this is Slurm job %s, and compute nodes have none"
                       % os.environ["SLURM_JOB_ID"])
    what = [w.strip() for w in (a.what or "").split(",") if w.strip()]
    bad = [w for w in what if w not in FETCH_WHAT]
    if not what or bad or len(set(what)) != len(what):
        raise CLIError("--what must list distinct kinds among %s, got %r" % (list(FETCH_WHAT), a.what))
    if ("taxonomy" in what) != bool(a.names_from):
        raise CLIError("--names-from goes with --what taxonomy, and taxonomy needs it")
    if ("known-items" in what) != bool(a.sources):
        raise CLIError("--sources goes with --what known-items, and known-items needs it")
    if a.slug and "refetch" not in what:
        raise CLIError("--slug goes with --what refetch")
    F = _module("fetch")
    out = {}
    for w in FETCH_WHAT:
        if w not in what:
            continue
        if w == "taxonomy":
            out[w] = ctx.call(F.taxonomy, ctx.domain, _file(a.names_from, "--names-from"),
                              ctx.out / TAXONOMY_NAME, None)
        elif w == "known-items":
            srcs = [_abs(s) for s in a.sources]
            missing = [str(s) for s in srcs if not s.exists()]
            if missing:
                raise CLIError("--sources %s do not exist" % missing)
            out[w] = ctx.call(F.known_items, srcs, ctx.out / KNOWN_ITEMS_NAME)
        elif w == "cards":
            out[w] = ctx.call(F.fetch_cards, ctx.domain, ctx.out, None)
        elif w == "kt7":
            out[w] = ctx.call(F.fetch_kt7, ctx.domain, ctx.out, None)
        else:
            slugs = list(a.slug or sorted((ctx.domain.raw.get("sources") or {}).get("card_image_counts") or {}))
            if not slugs:
                raise CLIError("--what refetch: no --slug and no sources.card_image_counts in the domain config")
            out[w] = {s: ctx.call(F.refetch, ctx.domain, s, ctx.out, None) for s in slugs}
    out["manifest"] = ctx.call(F.write_manifest, ctx.out)
    return out


HANDLERS = {"census": do_census, "leak": do_leak, "embed-judges": do_embed_judges, "qualify": do_qualify,
            "draw": do_draw, "sheets": do_sheets, "rl-b": do_rl_b, "ingest": do_ingest, "estimate": do_estimate,
            "map": do_map, "recover": do_recover, "fetch": do_fetch}


# ------------------------------------------------------------------ parser
class _Parser(argparse.ArgumentParser):
    """argparse, with its errors as CLIError (exit 2) instead of sys.exit."""

    def error(self, message):
        raise CLIError("%s: %s" % (self.prog, message))


def parser():
    ap = _Parser(prog=prog(), allow_abbrev=False,
                 description="The funnel audit: census, leak, judges, qualification, draw, sheets, labels, "
                             "estimates, maps, recovery and fetches (docs/FUNNEL_AUDIT_RUNNER.md).")
    common = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    common.add_argument("--prereg", required=True, help="the pre-registration (INC_DIR/funnel/prereg_v1.json); "
                                                        "the domain is the one it names")
    common.add_argument("--out", default=None, help="output directory (default %s; recover: %s)"
                                                    % (FUNNEL_DIR, R1_DIR))
    common.add_argument("--force", action="store_true", help="rebuild an output made from other inputs, where the "
                                                             "verb's function allows it")
    common.add_argument("--quiet", action="store_true")
    common.add_argument("--testing", action="store_true",
                        help="a test run: GPU verbs run without a CUDA device (fake models)")
    sub = ap.add_subparsers(dest="verb", required=True, parser_class=_Parser)
    p = {}
    for v in VERBS:
        if v == "sbatch-args":
            s = sub.add_parser(v, allow_abbrev=False, help="print the extra sbatch flags of a verb, one per line")
            s.add_argument("target", metavar="VERB")
        else:
            s = sub.add_parser(v, parents=[common], allow_abbrev=False)
        p[v] = s
    p["census"].add_argument("--taxonomy", default=None, help="taxonomy_cache.json (default <out>/%s)"
                                                              % TAXONOMY_NAME)
    p["census"].add_argument("--known-items", default=None, help="known_items_v1.json (default <out>/%s when it "
                                                                 "exists)" % KNOWN_ITEMS_NAME)
    p["census"].add_argument("--summaries-only", action="store_true",
                             help="only the funnel ledger from the Step 1 summaries and census_v0")
    p["census"].add_argument("--census-v0", default=None, help="--summaries-only: census_v0.json (default <out>/%s)"
                                                               % CENSUS_V0)
    p["embed-judges"].add_argument("--stage", choices=EMBED_STAGES, default="all")
    p["embed-judges"].add_argument("--shard", type=int, default=None)
    p["embed-judges"].add_argument("--nshards", type=int, default=None)
    p["embed-judges"].add_argument("--refetch", action="store_true",
                                   help="embed the refetch crop table (<out>/%s) instead" % REFETCH_TABLE)
    p["qualify"].add_argument("--rl", action="store_true", help="qualify the reference labellers from the gold "
                                                                "sentinels (F7) instead of the machine judges (F5)")
    p["rl-b"].add_argument("--endpoint", default=None, help="ollama endpoint (default $%s)" % ENDPOINT_ENV)
    p["rl-b"].add_argument("--model", default=None, help="the model tag (default and only allowed value: the "
                                                         "config's %s model)" % RL_B)
    p["rl-b"].add_argument("--sheet-dirs", nargs="+", default=None,
                           help="sheet directories (default <out>/%s and <out>/%s, those that exist)" % SHEET_DIRS)
    p["rl-b"].add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS)
    p["ingest"].add_argument("--answers", nargs="+", default=None,
                             help="answer directories (default every directory under <out>/%s)" % ANSWERS_DIR)
    p["map"].add_argument("--part", choices=MAP_PARTS, required=True)
    p["recover"].add_argument("--audit", default=None, help="<funnel dir>/%s" % AUDIT_NAME)
    p["recover"].add_argument("--maps", default=None, help="<funnel dir>/%s" % MAPS_NAME)
    p["recover"].add_argument("--policy", default=None, help="comma-separated, among %s" % ",".join(RECOVER_POLICIES))
    p["recover"].add_argument("--arms", action="store_true", help="build the realloop_v2 arm manifests")
    p["recover"].add_argument("--realloop", default=None, help="--arms: the realloop_v2 experiment (name or path)")
    p["recover"].add_argument("--base", default=None, help="--arms: base B's manifest")
    p["fetch"].add_argument("--what", required=True, help="comma-separated, among %s" % ",".join(FETCH_WHAT))
    p["fetch"].add_argument("--names-from", default=None, help="--what taxonomy: the file whose names are resolved")
    p["fetch"].add_argument("--sources", nargs="+", default=None, help="--what known-items: the source documents")
    p["fetch"].add_argument("--slug", nargs="+", default=None,
                            help="--what refetch: the sources to refetch (default: sources.card_image_counts)")
    return ap


# -------------------------------------------------------------------- main
def run(argv):
    a = parser().parse_args(argv)
    if a.verb == "sbatch-args":
        for line in sbatch_args(a.target):
            print(line)
        return 0
    cls = VERB_CLASS[a.verb]
    if cls in GPU_CLASSES and not a.testing and not cuda_available():
        extra = " ".join(SBATCH_RESOURCES[cls])
        raise CLIError("%s needs a GPU (class %s) and no CUDA device is visible: submit it as `sbatch $(%s "
                       "sbatch-args %s) %s %s ...`%s, or pass --testing with fake models"
                       % (a.verb, cls, prog(), a.verb, JOB_SCRIPT, a.verb,
                          " (that is: %s)" % extra if extra else ""))
    prereg_path = _abs(a.prereg)
    pre, dom = _module("domain").load_pair(prereg_path)
    out = _abs(a.out) if a.out else (R1_DIR if a.verb == "recover" else FUNNEL_DIR)
    out.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(a, prereg_path, pre, dom, out)
    log("start %s: prereg %s (core %s), domain %s, out %s"
        % (a.verb, prereg_path, pre.core_sha256[:12], dom.name, out), quiet=a.quiet)
    result = HANDLERS[a.verb](ctx)
    print("[funnel] %s: %s" % (a.verb, _summary(result, out)), flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        return run(argv)
    except SystemExit as e:                      # --help
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except FunnelError as e:
        print("[funnel] refused: %s" % e, file=sys.stderr, flush=True)
        return 2
    except Exception:                            # noqa: BLE001 - a crash is exit 1, with its traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
