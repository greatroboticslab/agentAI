#!/usr/bin/env python3
"""The funnel CLI, funnel/__main__.py (docs/FUNNEL_AUDIT_RUNNER.md 5.6.1).

Every engine module the CLI calls is replaced by a recording fake installed in
sys.modules under its package name (the CLI imports each one when its verb
runs), so this test pins the CLI itself: what each verb calls, with which
arguments, and how it refuses. Then, when the real domain config exists, the
real domain loader is used with the committed prereg and contract (copied
into a temporary REPO).

Pinned:
  * the tables: VERBS (13), VERB_CLASS (every verb but sbatch-args; classes
    cpu, gpu, gpu_large, lab), SBATCH_RESOURCES as the runner gives it;
  * sbatch-args prints SBATCH_RESOURCES[VERB_CLASS[VERB]] one per line for
    every job verb, and refuses fetch (lab), itself and an unknown verb;
  * argparse: every verb parses its options; --domain is refused (exit 2) on
    every verb; a missing --prereg, no verb, an unknown verb or an option of
    another verb exits 2; --help exits 0; abbreviations are refused;
  * every verb reaches its function with the pinned arguments (a parameter
    named prereg gets the loaded prereg, prereg_path its path, domain the
    loaded config, adapter the module adapters.load returned), and prints one
    closing line "[funnel] <verb>: ..." last on stdout;
  * the verbs' own rules: census needs the taxonomy cache (the refusal names
    lever L12) and passes the known-items file only when it exists;
    --summaries-only calls ledger_from_summaries; embed-judges runs every
    shard, one --shard, the judges, or all, and --refetch embeds the refetch
    table with DINOv2 and with the adapter's one *_embedder; rl-b needs an
    endpoint, uses the configured model only, checks vision and answers the
    sheet directories that exist; ingest reads every answer directory;
    estimate writes audit_v1.json when the evaluator did not (and replaces a
    stale one) and audit_v1.md from render_md; recover takes audit_v1.json
    and class_maps.json from one directory, distinct known policies, writes
    to R1_DIR by default, and --arms takes an experiment name or path; fetch
    refuses inside a Slurm job, pairs --names-from with taxonomy and --sources
    with known-items, runs the kinds in FETCH_WHAT order and writes the
    manifest; refetch defaults to the config's card_image_counts sources;
  * GPU verbs refuse without a CUDA device (exit 2, printing the sbatch line
    to use) unless --testing;
  * --force reaches a function with a force parameter and is refused by one
    without; --testing and --quiet reach functions that take them;
  * exit codes: a FunnelError subclass -> 2 with the message on stderr; any
    other exception -> 1 with the traceback; a missing engine module -> 2;
  * estimate with an evaluator that writes its own files (as the real one
    does): a rerun that reproduces the audit keeps audit_v1.json's and
    audit_v1.md's bytes; a changed audit replaces them; an invalid audit exits
    2 and stays written;
  * every verb against the REAL engine modules and adapter: each call binds to
    the real function's signature;
  * the file names no domain term (fixed list of runner 7.5, and the terms of
    every domain config present).

No network, no GPU.

Run:  python3 tests/test_funnel_cli.py
"""
import contextlib
import io
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import types

import funnel_prereg as FPR

TMP = pathlib.Path(tempfile.mkdtemp(prefix="funnel_cli_"))
os.environ["INC_DIR"] = str(TMP / "inc")
os.environ["REPO"] = str(TMP / "repo")
for v in ("SLURM_JOB_ID", "FUNNEL_OLLAMA_ENDPOINT", "FUNNEL_CONTRACT", "SLURM_ARRAY_TASK_ID"):
    os.environ.pop(v, None)
ROOT = pathlib.Path(__file__).resolve().parents[1]           # the nested package root
GIT = ROOT.parent
sys.path.insert(0, str(ROOT))

CONTRACT = GIT / "docs" / "FUNNEL_AUDIT.md"
PREREG = ROOT / "results" / "framework" / "inc" / "funnel" / "prereg_v1.json"
(TMP / "repo" / "docs").mkdir(parents=True)
shutil.copyfile(CONTRACT, TMP / "repo" / "docs" / "FUNNEL_AUDIT.md")
PREREG_COPY = TMP / "inc" / "funnel" / "prereg_v1.json"
PREREG_COPY.parent.mkdir(parents=True)
FPR.write_pre_draw(PREREG_COPY, PREREG)

from weed_optimizer_framework.tools import funnel as F  # noqa: E402
from weed_optimizer_framework.tools.funnel import __main__ as M  # noqa: E402

PKG = F.__name__
FAILURES = []
SKIPS = []
CALLS = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def run(argv):
    """(exit code, stdout, stderr) of M.main(argv)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = M.main(list(argv))
    return rc, out.getvalue(), err.getvalue()


def last_line(text):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# --------------------------------------------------------------- the fakes
class FakePrereg(object):
    detector = 1                      # the copy detector version it requires (a test sets 2: amendment A2)

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.core_sha256 = "c" * 64
        self.domain_name = "fakedom"
        self.detector_version = FakePrereg.detector


class FakeDomain(object):
    name = "fakedom"
    adapter = "fake_step1"
    raw = {"reference_labeller": {"backends": {"RL-B": {"model": "vlm:7b"}, "RL-A": {"model": "ext"}}},
           "sources": {"card_image_counts": {"s_b": 10, "s_a": 5}},
           "judges": {"features": {"model": "vision/base", "pooling": "cls"}}}


PRE_OBJ, DOM_OBJ = [], []


def rec(name, ret=None):
    """A recording function named `name`: it appends (name, args, kwargs)."""
    def fn(*args, **kwargs):
        CALLS.append((name, args, kwargs))
        return {"format": "funnel-fake/1", "fn": name} if ret is None else ret(*args, **kwargs)
    return fn


def signature_fn(name, params, ret=None):
    """A recording function with an explicit signature (so the CLI's keyword
    filtering sees the parameters the runner pins)."""
    src = "def %s(%s):\n    return _rec(%s)\n" % (
        name.replace(".", "_").replace("-", "_"), ", ".join(params),
        ", ".join("%s=%s" % (p.split("=")[0], p.split("=")[0]) for p in params))
    ns = {}

    def _rec(**kw):
        CALLS.append((name, (), kw))
        return {"format": "funnel-fake/1", "fn": name} if ret is None else ret(**kw)
    ns["_rec"] = _rec
    exec(src, ns)
    return ns[name.replace(".", "_").replace("-", "_")]


def load_pair(path):
    CALLS.append(("domain.load_pair", (path,), {}))
    if not pathlib.Path(path).is_file():
        raise F.PreregError("no prereg at %s" % path)
    pre, dom = FakePrereg(path), FakeDomain()
    PRE_OBJ[:] = [pre]
    DOM_OBJ[:] = [dom]
    return pre, dom


class FakeEmbedder(object):
    def __init__(self, name):
        self.name = name


class FakeClient(object):
    def __init__(self, endpoint, model, transport=None, num_ctx=8192):
        CALLS.append(("rl.OllamaClient", (endpoint, model), {}))
        self.endpoint, self.model = endpoint, model

    def check_vision(self):
        CALLS.append(("rl.check_vision", (self.model,), {}))


RAISE = {}          # function name -> exception to raise


def maybe_raise(name):
    if name in RAISE:
        raise RAISE[name]


def make_fakes():
    mods = {}

    def mod(name, **attrs):
        m = types.ModuleType("%s.%s" % (PKG, name))
        for k, v in attrs.items():
            setattr(m, k, v)
        mods[name] = m
        return m

    adapter = types.SimpleNamespace(
        census=signature_fn("adapter.census", ["prereg", "domain", "out_dir", "taxonomy_cache", "known_items=None",
                                               "step1_dir=None", "force=False"]),
        ledger_from_summaries=signature_fn("adapter.ledger_from_summaries",
                                           ["domain", "step1_dir", "census_v0", "out_path",
                                            "fit_info_projection=None"]),
        crop_table=rec("adapter.crop_table", lambda: "CROPS"),
        thing_embedder=rec("adapter.thing_embedder", lambda: FakeEmbedder("thing:model")))
    mod("domain", load_pair=load_pair, leak_detector_version=lambda pre: pre.detector_version)
    mod("adapters", INTERFACE=("census", "ledger_from_summaries", "crop_table", "thing_embedder", "text_encoder"),
        load=rec("adapters.load", lambda name: adapter))

    def leak_run(prereg, domain, funnel_dir, adapter, embedder=None):
        maybe_raise("leak.run")
        CALLS.append(("leak.run", (prereg, domain, funnel_dir, adapter), {"embedder": embedder}))
        return {"format": "funnel-leak/1", "calibration_ok": True}

    def leak_run_v2(prereg, domain, funnel_dir, adapter, embedder=None):
        CALLS.append(("leak.run_v2", (prereg, domain, funnel_dir, adapter), {"embedder": embedder}))
        return {"format": "funnel-leak/2", "detector_version": 2}
    mod("leak", run=leak_run, run_v2=leak_run_v2)
    mod("embed", features_config=lambda d: ("vision/base", "cls"),
        Dinov2Embedder=lambda model, pooling: FakeEmbedder("%s:%s" % (model, pooling)),
        embed_crops=signature_fn("embed.embed_crops", ["crop_table", "out_dir", "shard", "nshards", "embedder=None",
                                                       "procs=5", "batch=64", "chunk_images=2000", "force=False",
                                                       "model=None", "pooling='cls'"],
                                 lambda **kw: {"crops": 7, "embedder": "%s:%s" % (kw["model"], kw["pooling"]),
                                               "crops_sha256": "x"}),
        embed_table=signature_fn("embed.embed_table", ["csv_path", "out_path", "embedder", "procs=1", "batch=64",
                                                       "force=False"]))
    mod("judges", score_all=signature_fn("judges.score_all", ["prereg", "domain", "funnel_dir", "adapter",
                                                              "text_encoder=None", "force=False", "testing=False"]))
    mod("qualify", judges=signature_fn("qualify.judges", ["prereg", "domain", "funnel_dir", "adapter"]),
        rl=signature_fn("qualify.rl", ["prereg", "domain", "funnel_dir"]))

    def draw_fn(prereg_path, out_dir, adapter, force=False):
        maybe_raise("draw.draw")
        CALLS.append(("draw.draw", (prereg_path, out_dir, adapter), {"force": force}))
        return {"format": "funnel-frames/1", "groups": {}}
    mod("draw", draw=draw_fn)
    mod("sheets", run=signature_fn("sheets.run", ["prereg", "domain", "funnel_dir", "adapter"]))
    mod("rl", OllamaClient=FakeClient,
        run_rl_b=signature_fn("rl.run_rl_b", ["prereg", "domain", "funnel_dir", "client", "sheet_dirs",
                                              "max_attempts=3"]),
        ingest=signature_fn("rl.ingest", ["prereg", "domain", "funnel_dir", "answer_dirs"]))
    mod("estimate", evaluate=signature_fn("estimate.evaluate", ["prereg_path", "funnel_dir", "adapter"],
                                          lambda **kw: {"format": "funnel-audit/1", "valid": True,
                                                        "built_utc": "2026-09-28T00:00:00Z", "hypotheses": {}}),
        render_md=lambda audit: "# audit\n\nvalid: %s\n" % audit["valid"])
    mod("relation", run_geometry=signature_fn("relation.run_geometry", ["prereg", "domain", "funnel_dir", "adapter"]),
        run_relation=signature_fn("relation.run_relation", ["prereg", "domain", "funnel_dir", "adapter"]))
    mod("recover", run=signature_fn("recover.run", ["prereg", "domain", "funnel_dir", "out_dir", "policies",
                                                    "adapter", "audit_path=None", "maps_path=None"]),
        arms=signature_fn("recover.arms", ["realloop_exp_dir", "out_dir", "base_manifest", "adapter"]))
    mod("fetch", taxonomy=signature_fn("fetch.taxonomy", ["domain", "names_from", "out_path", "transport"]),
        known_items=signature_fn("fetch.known_items", ["sources", "out_path"]),
        fetch_cards=signature_fn("fetch.fetch_cards", ["domain", "out_dir", "transport"]),
        fetch_kt7=signature_fn("fetch.fetch_kt7", ["domain", "out_dir", "transport"]),
        refetch=signature_fn("fetch.refetch", ["domain", "slug", "out_dir", "transport"]),
        write_manifest=signature_fn("fetch.write_manifest", ["out_dir", "prereg=None", "testing=False"]))
    return mods, adapter


class Fakes(object):
    """sys.modules holds the fakes inside the block; the real modules (if any) come back after."""

    def __enter__(self):
        self.mods, self.adapter = make_fakes()
        self.saved = {n: sys.modules.get("%s.%s" % (PKG, n)) for n in self.mods}
        for n, m in self.mods.items():
            sys.modules["%s.%s" % (PKG, n)] = m
        return self

    def __exit__(self, *exc):
        for n, m in self.saved.items():
            if m is None:
                sys.modules.pop("%s.%s" % (PKG, n), None)
            else:
                sys.modules["%s.%s" % (PKG, n)] = m
        return False


def calls(name):
    return [c for c in CALLS if c[0] == name]


def one(name):
    got = calls(name)
    return got[0] if len(got) == 1 else None


# ------------------------------------------------------------------ tables
def test_tables():
    print("tables")
    check("VERBS: the 13 verbs of the runner, in its order",
          M.VERBS == ("census", "leak", "embed-judges", "qualify", "draw", "sheets", "rl-b", "ingest", "estimate",
                      "map", "recover", "fetch", "sbatch-args"))
    check("VERB_CLASS: every verb but sbatch-args, classes as the runner gives them",
          M.VERB_CLASS == {"census": "cpu", "leak": "gpu", "embed-judges": "gpu", "qualify": "cpu", "draw": "cpu",
                           "sheets": "cpu", "rl-b": "gpu_large", "ingest": "cpu", "estimate": "cpu", "map": "cpu",
                           "recover": "cpu", "fetch": "lab"}
          and set(M.VERB_CLASS) == set(M.VERBS) - {"sbatch-args"})
    check("SBATCH_RESOURCES: nothing extra for cpu and gpu (the script's own lines), an H100-80 for gpu_large",
          M.SBATCH_RESOURCES == {"cpu": [], "gpu": [],
                                 "gpu_large": ["--gres=gpu:h100-80:1", "--cpus-per-task=12", "--mem=80G"]})
    check("the verb handlers cover every verb but sbatch-args", set(M.HANDLERS) == set(M.VERB_CLASS))
    good = True
    for v, cls in sorted(M.VERB_CLASS.items()):
        rc, out, err = run(["sbatch-args", v])
        if cls == "lab":
            good = good and rc == 2 and "lab" in err and out == ""
        else:
            good = good and rc == 0 and out.splitlines() == M.SBATCH_RESOURCES[cls]
    check("sbatch-args prints SBATCH_RESOURCES[VERB_CLASS[VERB]] one per line; fetch (lab) is refused", good)
    check("sbatch-args refuses an unknown verb and itself",
          run(["sbatch-args", "bogus"])[0] == 2 and run(["sbatch-args", "sbatch-args"])[0] == 2)


# ------------------------------------------------------------------ parser
EXAMPLES = {
    "census": ["--taxonomy", "t.json", "--known-items", "k.json"],
    "leak": [],
    "embed-judges": ["--stage", "embed", "--shard", "1", "--nshards", "4"],
    "qualify": ["--rl"],
    "draw": [],
    "sheets": [],
    "rl-b": ["--endpoint", "http://x", "--model", "m", "--sheet-dirs", "a", "b", "--max-attempts", "2"],
    "ingest": ["--answers", "a", "b"],
    "estimate": [],
    "map": ["--part", "geometry"],
    "recover": ["--audit", "a.json", "--maps", "m.json", "--policy", "R-A,R-V"],
    "fetch": ["--what", "cards,kt7"],
}


def test_parser():
    print("argparse")
    ap = M.parser()
    ok = True
    for v, extra in EXAMPLES.items():
        try:
            a = ap.parse_args([v, "--prereg", "p.json", "--out", "o", "--force", "--quiet", "--testing"] + extra)
            ok = ok and a.verb == v and a.prereg == "p.json" and a.force and a.quiet and a.testing
        except F.CLIError as e:
            ok = False
            print("       %s: %s" % (v, e))
    check("every verb parses its options with the common ones", ok)
    refused = [v for v in EXAMPLES if run([v, "--prereg", "p.json", "--domain", "x"] + EXAMPLES[v])[0] == 2]
    check("--domain is refused on every verb (exit 2): the domain comes from the prereg",
          sorted(refused) == sorted(EXAMPLES), sorted(set(EXAMPLES) - set(refused)))
    rc, _o, err = run(["census", "--domain", "x", "--prereg", "p.json"])
    check("... with argparse's message on stderr", rc == 2 and "unrecognized arguments: --domain" in err, err)
    check("a missing --prereg, no verb, an unknown verb, another verb's option: exit 2",
          run(["leak"])[0] == 2 and run([])[0] == 2 and run(["bogus", "--prereg", "p"])[0] == 2
          and run(["leak", "--prereg", "p", "--part", "geometry"])[0] == 2)
    check("abbreviated options are refused (--pre for --prereg)", run(["leak", "--pre", "p"])[0] == 2)
    check("--help exits 0", run(["--help"])[0] == 0 and run(["census", "--help"])[0] == 0)
    check("choices: --stage, --part and a non-integer --shard are checked by argparse (exit 2)",
          run(["embed-judges", "--prereg", "p", "--stage", "x"])[0] == 2
          and run(["map", "--prereg", "p", "--part", "x"])[0] == 2
          and run(["embed-judges", "--prereg", "p", "--shard", "a"])[0] == 2)


# ---------------------------------------------------------------- dispatch
def fresh_out(name):
    d = TMP / "out" / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def test_dispatch():
    print("every verb reaches its function")
    pre = str(PREREG_COPY)
    saved_cuda = M.cuda_available
    M.cuda_available = lambda: False
    try:
        with Fakes() as fk:
            ad = fk.adapter

            # census
            out = fresh_out("census")
            CALLS[:] = []
            rc, so, se = run(["census", "--prereg", pre, "--out", str(out)])
            check("census without the taxonomy cache refuses, naming lever L12 (exit 2)",
                  rc == 2 and "run fetch --what taxonomy (lever L12)" in se and "no resolution" in se, se)
            (out / "taxonomy_cache.json").write_text("{}")
            CALLS[:] = []
            rc, so, se = run(["census", "--prereg", pre, "--out", str(out)])
            c = one("adapter.census")
            check("census calls adapter.census(prereg, domain, out, taxonomy cache, known_items=None when "
                  "known_items_v1.json is absent) with the loaded prereg and domain; one closing line",
                  rc == 0 and c is not None and c[2]["prereg"] is PRE_OBJ[0] and c[2]["domain"] is DOM_OBJ[0]
                  and c[2]["out_dir"] == out and c[2]["taxonomy_cache"] == out / "taxonomy_cache.json"
                  and c[2]["known_items"] is None and c[2]["force"] is False
                  and last_line(so).startswith("[funnel] census: ") and so.count("[funnel] census:") == 1
                  and one("adapters.load")[1] == ("fake_step1",), (rc, se, c))
            (out / "known_items_v1.json").write_text("{}")
            CALLS[:] = []
            rc, so, se = run(["census", "--prereg", pre, "--out", str(out), "--force"])
            c = one("adapter.census")
            check("census passes known_items_v1.json when it exists, and --force",
                  rc == 0 and c[2]["known_items"] == out / "known_items_v1.json" and c[2]["force"] is True)
            rc, so, se = run(["census", "--prereg", pre, "--out", str(out), "--known-items", str(TMP / "none.json")])
            check("census refuses an explicit --known-items that does not exist", rc == 2, se)
            CALLS[:] = []
            v0 = out / "census_v0.json"
            v0.write_text("{}")
            rc, so, se = run(["census", "--prereg", pre, "--out", str(out), "--summaries-only"])
            c = one("adapter.ledger_from_summaries")
            check("census --summaries-only calls ledger_from_summaries(domain, STEP1_DIR, census_v0, "
                  "<out>/funnel_ledger.json)",
                  rc == 0 and c[2]["domain"] is DOM_OBJ[0] and c[2]["step1_dir"] == F.STEP1_DIR
                  and c[2]["census_v0"] == v0 and c[2]["out_path"] == out / "funnel_ledger.json"
                  and not calls("adapter.census"), (rc, se))
            check("census: --census-v0 without --summaries-only, and --taxonomy with it, refuse",
                  run(["census", "--prereg", pre, "--out", str(out), "--census-v0", str(v0)])[0] == 2
                  and run(["census", "--prereg", pre, "--out", str(out), "--summaries-only", "--taxonomy",
                           "t"])[0] == 2)

            # leak and the GPU rule
            out = fresh_out("leak")
            CALLS[:] = []
            rc, so, se = run(["leak", "--prereg", pre, "--out", str(out)])
            check("a GPU verb without a CUDA device refuses (exit 2) and prints the sbatch line to use",
                  rc == 2 and not calls("leak.run") and "sbatch $(python -m %s sbatch-args leak) run_inc_funnel.sh "
                  "leak" % PKG in se, se)
            rc, so, se = run(["rl-b", "--prereg", pre, "--out", str(out)])
            check("... rl-b's line carries the H100-80 flags",
                  rc == 2 and "--gres=gpu:h100-80:1 --cpus-per-task=12 --mem=80G" in se, se)
            CALLS[:] = []
            rc, so, se = run(["leak", "--prereg", pre, "--out", str(out), "--testing"])
            c = one("leak.run")
            check("leak --testing runs without a device: leak.run(prereg, domain, out, adapter)",
                  rc == 0 and c[1] == (PRE_OBJ[0], DOM_OBJ[0], out, ad) and c[2] == {"embedder": None}, (rc, se))
            M.cuda_available = lambda: True
            CALLS[:] = []
            rc, so, se = run(["leak", "--prereg", pre, "--out", str(out)])
            check("leak with a CUDA device runs without --testing", rc == 0 and one("leak.run") is not None)
            M.cuda_available = lambda: False
            FakePrereg.detector = 2
            CALLS[:] = []
            rc, so, se = run(["leak", "--prereg", pre, "--out", str(out), "--testing"])
            c = one("leak.run_v2")
            check("a prereg whose amendment requires copy detector version 2: leak runs leak.run_v2 (leak_v2.json), "
                  "never leak.run", rc == 0 and c is not None and c[1] == (PRE_OBJ[0], DOM_OBJ[0], out, ad)
                  and not calls("leak.run"), (rc, se))
            FakePrereg.detector = 3
            rc, so, se = run(["leak", "--prereg", pre, "--out", str(out), "--testing"])
            check("  a version the engine does not implement is refused (exit 2)", rc == 2 and "version 3" in se, se)
            FakePrereg.detector = 1

            # embed-judges
            out = fresh_out("embed")
            CALLS[:] = []
            rc, so, se = run(["embed-judges", "--prereg", pre, "--out", str(out), "--testing", "--stage", "embed",
                              "--nshards", "3"])
            ec = calls("embed.embed_crops")
            check("embed-judges --stage embed --nshards 3: every shard 0..2 of 3, the adapter's crop table, "
                  "<out>/emb_dinov2, the config's model and pooling; no judges",
                  rc == 0 and [(c[2]["shard"], c[2]["nshards"]) for c in ec] == [(0, 3), (1, 3), (2, 3)]
                  and all(c[2]["crop_table"] == "CROPS" and c[2]["out_dir"] == out / "emb_dinov2"
                          and c[2]["model"] == "vision/base" and c[2]["pooling"] == "cls" for c in ec)
                  and not calls("judges.score_all"), (rc, se))
            CALLS[:] = []
            rc, so, se = run(["embed-judges", "--prereg", pre, "--out", str(out), "--testing", "--stage", "embed",
                              "--shard", "1", "--nshards", "4", "--force"])
            ec = calls("embed.embed_crops")
            check("--shard 1 --nshards 4: that shard only, with --force",
                  rc == 0 and [(c[2]["shard"], c[2]["nshards"], c[2]["force"]) for c in ec] == [(1, 4, True)])
            CALLS[:] = []
            rc, so, se = run(["embed-judges", "--prereg", pre, "--out", str(out), "--testing", "--stage", "judges"])
            c = one("judges.score_all")
            check("--stage judges: judges.score_all(prereg, domain, out, adapter, testing=True) only",
                  rc == 0 and c[2]["prereg"] is PRE_OBJ[0] and c[2]["adapter"] is ad and c[2]["testing"] is True
                  and c[2]["text_encoder"] is None and not calls("embed.embed_crops"), (rc, se))
            CALLS[:] = []
            rc, so, se = run(["embed-judges", "--prereg", pre, "--out", str(out), "--testing"])
            check("the default stage 'all': one shard of 1, then the judges",
                  rc == 0 and [(c[2]["shard"], c[2]["nshards"]) for c in calls("embed.embed_crops")] == [(0, 1)]
                  and one("judges.score_all") is not None)
            check("--shard without --nshards, and --stage judges with shards, refuse",
                  run(["embed-judges", "--prereg", pre, "--out", str(out), "--testing", "--shard", "1"])[0] == 2
                  and run(["embed-judges", "--prereg", pre, "--out", str(out), "--testing", "--stage", "judges",
                           "--nshards", "2"])[0] == 2)
            rc, so, se = run(["embed-judges", "--prereg", pre, "--out", str(out), "--testing", "--stage", "embed",
                              "--refetch"])
            check("--refetch without the refetch crop table refuses", rc == 2 and "refetch crop table" in se, se)
            (out / "refetch").mkdir()
            (out / "refetch" / "crops_refetch.csv").write_text("x\n")
            CALLS[:] = []
            rc, so, se = run(["embed-judges", "--prereg", pre, "--out", str(out), "--testing", "--stage", "embed",
                              "--refetch"])
            et = calls("embed.embed_table")
            check("--refetch embeds the refetch table with DINOv2 and with the adapter's one *_embedder",
                  rc == 0 and [(c[2]["csv_path"].name, c[2]["out_path"].name, c[2]["embedder"].name) for c in et]
                  == [("crops_refetch.csv", "emb_dinov2_refetch.npz", "vision/base:cls"),
                      ("crops_refetch.csv", "emb_thing_refetch.npz", "thing:model")]
                  and not calls("embed.embed_crops"), (rc, se, et))
            check("--refetch with --stage judges or with shards refuses",
                  run(["embed-judges", "--prereg", pre, "--out", str(out), "--testing", "--refetch"])[0] == 2
                  and run(["embed-judges", "--prereg", pre, "--out", str(out), "--testing", "--stage", "embed",
                           "--refetch", "--nshards", "2"])[0] == 2)

            # qualify, draw, sheets, map
            out = fresh_out("misc")
            CALLS[:] = []
            r1 = run(["qualify", "--prereg", pre, "--out", str(out)])
            r2 = run(["qualify", "--prereg", pre, "--out", str(out), "--rl"])
            check("qualify calls qualify.judges(prereg, domain, out, adapter); --rl calls qualify.rl(prereg, "
                  "domain, out)",
                  r1[0] == 0 and r2[0] == 0 and one("qualify.judges")[2]["adapter"] is ad
                  and set(one("qualify.rl")[2]) == {"prereg", "domain", "funnel_dir"})
            CALLS[:] = []
            rc, so, se = run(["draw", "--prereg", pre, "--out", str(out), "--force"])
            c = one("draw.draw")
            check("draw calls draw.draw(prereg path, out, adapter), and --force reaches it",
                  rc == 0 and c[1] == (PREREG_COPY, out, ad) and c[2] == {"force": True})
            CALLS[:] = []
            rc, so, se = run(["sheets", "--prereg", pre, "--out", str(out)])
            check("sheets calls sheets.run(prereg, domain, out, adapter)",
                  rc == 0 and one("sheets.run")[2] == {"prereg": PRE_OBJ[0], "domain": DOM_OBJ[0], "funnel_dir": out,
                                                       "adapter": ad})
            rc, so, se = run(["sheets", "--prereg", pre, "--out", str(out), "--force"])
            check("--force given to a function without a force parameter is refused, never dropped",
                  rc == 2 and "has no force parameter" in se, se)
            CALLS[:] = []
            run(["map", "--prereg", pre, "--out", str(out), "--part", "geometry"])
            run(["map", "--prereg", pre, "--out", str(out), "--part", "relation"])
            check("map --part geometry / relation call relation.run_geometry / run_relation",
                  one("relation.run_geometry") is not None and one("relation.run_relation") is not None)

            # rl-b
            out = fresh_out("rlb")
            rc, so, se = run(["rl-b", "--prereg", pre, "--out", str(out), "--testing"])
            check("rl-b without --endpoint or $FUNNEL_OLLAMA_ENDPOINT refuses", rc == 2 and "--endpoint" in se, se)
            os.environ["FUNNEL_OLLAMA_ENDPOINT"] = "http://127.0.0.1:8123"
            try:
                rc, so, se = run(["rl-b", "--prereg", pre, "--out", str(out), "--testing"])
                check("rl-b with no sheet directory refuses", rc == 2 and "sheet directory" in se, se)
                (out / "sheets_v1").mkdir()
                (out / "sheets_v1_cluster").mkdir()
                CALLS[:] = []
                rc, so, se = run(["rl-b", "--prereg", pre, "--out", str(out), "--testing"])
                c = one("rl.run_rl_b")
                check("rl-b: OllamaClient($FUNNEL_OLLAMA_ENDPOINT, the configured model), check_vision, then "
                      "run_rl_b on both sheet directories with 3 attempts",
                      rc == 0 and one("rl.OllamaClient")[1] == ("http://127.0.0.1:8123", "vlm:7b")
                      and one("rl.check_vision") is not None
                      and c[2]["sheet_dirs"] == [out / "sheets_v1", out / "sheets_v1_cluster"]
                      and c[2]["max_attempts"] == 3 and c[2]["client"].model == "vlm:7b"
                      and [x[0] for x in CALLS].index("rl.check_vision")
                      < [x[0] for x in CALLS].index("rl.run_rl_b"), (rc, se))
                CALLS[:] = []
                rc, so, se = run(["rl-b", "--prereg", pre, "--out", str(out), "--testing", "--endpoint", "http://e",
                                  "--sheet-dirs", str(out / "sheets_v1"), "--max-attempts", "5"])
                check("rl-b: --endpoint, --sheet-dirs and --max-attempts override the defaults",
                      rc == 0 and one("rl.OllamaClient")[1][0] == "http://e"
                      and one("rl.run_rl_b")[2]["sheet_dirs"] == [out / "sheets_v1"]
                      and one("rl.run_rl_b")[2]["max_attempts"] == 5)
                check("rl-b refuses another model tag, a missing sheet directory and --max-attempts 0",
                      run(["rl-b", "--prereg", pre, "--out", str(out), "--testing", "--model", "other"])[0] == 2
                      and run(["rl-b", "--prereg", pre, "--out", str(out), "--testing", "--sheet-dirs",
                               str(out / "nope")])[0] == 2
                      and run(["rl-b", "--prereg", pre, "--out", str(out), "--testing", "--max-attempts", "0"])[0] == 2)
            finally:
                os.environ.pop("FUNNEL_OLLAMA_ENDPOINT", None)

            # ingest
            out = fresh_out("ingest")
            check("ingest with no answer directory refuses",
                  run(["ingest", "--prereg", pre, "--out", str(out)])[0] == 2)
            for b in ("RL-B", "RL-A"):
                (out / "rl_answers" / b).mkdir(parents=True)
            CALLS[:] = []
            rc, so, se = run(["ingest", "--prereg", pre, "--out", str(out)])
            check("ingest reads every directory under <out>/rl_answers, sorted",
                  rc == 0 and one("rl.ingest")[2]["answer_dirs"] == [out / "rl_answers" / "RL-A",
                                                                     out / "rl_answers" / "RL-B"])

            # estimate
            out = fresh_out("estimate")
            (out / "audit_v1.json").write_text(json.dumps({"format": "funnel-audit/1", "valid": False}))
            CALLS[:] = []
            rc, so, se = run(["estimate", "--prereg", pre, "--out", str(out)])
            aj = json.loads((out / "audit_v1.json").read_text())
            check("estimate calls estimate.evaluate(prereg path, out, adapter), replaces a stale audit_v1.json with "
                  "the returned audit and writes audit_v1.md from render_md",
                  rc == 0 and one("estimate.evaluate")[2] == {"prereg_path": PREREG_COPY, "funnel_dir": out,
                                                              "adapter": ad}
                  and aj["valid"] is True and (out / "audit_v1.md").read_text() == "# audit\n\nvalid: True\n"
                  and not list(out.glob("*.tmp")), (rc, se))
            before = (out / "audit_v1.json").stat().st_mtime_ns
            ino = (out / "audit_v1.json").stat().st_ino
            rc, so, se = run(["estimate", "--prereg", pre, "--out", str(out)])
            check("estimate leaves an audit_v1.json that already holds the audit (volatile keys aside) untouched",
                  rc == 0 and (out / "audit_v1.json").stat().st_ino == ino
                  and (out / "audit_v1.json").stat().st_mtime_ns == before)

            # recover
            fun = fresh_out("recover_funnel")
            (fun / "audit_v1.json").write_text("{}")
            (fun / "class_maps.json").write_text("{}")
            CALLS[:] = []
            rc, so, se = run(["recover", "--prereg", pre, "--audit", str(fun / "audit_v1.json"), "--maps",
                              str(fun / "class_maps.json"), "--policy", "R-A,R-C,R-T,R-V,R-J"])
            c = one("recover.run")
            check("recover calls recover.run(prereg, domain, <the audit's directory>, R1_DIR by default, "
                  "[policies], adapter), with audit_path and maps_path where it takes them",
                  rc == 0 and c[2]["funnel_dir"] == fun and c[2]["out_dir"] == F.R1_DIR
                  and c[2]["policies"] == ["R-A", "R-C", "R-T", "R-V", "R-J"] and c[2]["adapter"] is ad
                  and c[2]["audit_path"] == fun / "audit_v1.json" and c[2]["maps_path"] == fun / "class_maps.json"
                  and F.R1_DIR.is_dir(), (rc, se))
            other = fresh_out("recover_other")
            (other / "class_maps.json").write_text("{}")
            (other / "audit.json").write_text("{}")
            bad = [
                ["--audit", str(fun / "audit_v1.json"), "--maps", str(other / "class_maps.json"), "--policy", "R-A"],
                ["--audit", str(other / "audit.json"), "--maps", str(other / "class_maps.json"), "--policy", "R-A"],
                ["--audit", str(fun / "audit_v1.json"), "--maps", str(fun / "class_maps.json"), "--policy", "R-X"],
                ["--audit", str(fun / "audit_v1.json"), "--maps", str(fun / "class_maps.json"), "--policy", "R-A,R-A"],
                ["--audit", str(fun / "audit_v1.json"), "--maps", str(fun / "class_maps.json")],
                ["--audit", str(fun / "audit_v1.json"), "--maps", str(fun / "class_maps.json"), "--policy", "R-A",
                 "--base", "x"]]
            check("recover refuses: files from two directories, another file name, an unknown or repeated policy, "
                  "no policy, --base without --arms",
                  all(run(["recover", "--prereg", pre] + b)[0] == 2 for b in bad))
            exp = pathlib.Path(os.environ["INC_DIR"]) / "realloop_vX"
            exp.mkdir(parents=True)
            (exp / "exp.json").write_text("{}")
            base = TMP / "base_B.jsonl"
            base.write_text("")
            CALLS[:] = []
            rc, so, se = run(["recover", "--prereg", pre, "--arms", "--realloop", "realloop_vX", "--base", str(base)])
            rc2, _s, _e = run(["recover", "--prereg", pre, "--arms", "--realloop", str(exp), "--base", str(base)])
            a1, a2 = calls("recover.arms")
            check("recover --arms calls recover.arms(<the experiment dir>, R1_DIR, base, adapter), from a name under "
                  "INC_DIR or a path",
                  rc == 0 and rc2 == 0 and a1[2] == a2[2] == {"realloop_exp_dir": exp, "out_dir": F.R1_DIR,
                                                              "base_manifest": base, "adapter": ad}, (rc, se))
            check("recover --arms refuses --policy, an unknown experiment and a missing --base",
                  run(["recover", "--prereg", pre, "--arms", "--realloop", "realloop_vX", "--base", str(base),
                       "--policy", "R-A"])[0] == 2
                  and run(["recover", "--prereg", pre, "--arms", "--realloop", "nope", "--base", str(base)])[0] == 2
                  and run(["recover", "--prereg", pre, "--arms", "--realloop", "realloop_vX"])[0] == 2)

            # fetch
            out = fresh_out("fetch")
            names = TMP / "pool_summary.json"
            names.write_text("{}")
            docs = TMP / "literature"
            docs.mkdir()
            os.environ["SLURM_JOB_ID"] = "4242"
            try:
                rc, so, se = run(["fetch", "--prereg", pre, "--out", str(out), "--what", "cards"])
                check("fetch refuses inside a Slurm job (the lab fetches)", rc == 2 and "Slurm job 4242" in se, se)
            finally:
                os.environ.pop("SLURM_JOB_ID")
            CALLS[:] = []
            rc, so, se = run(["fetch", "--prereg", pre, "--out", str(out), "--what", "refetch,kt7,cards,known-items,"
                              "taxonomy", "--names-from", str(names), "--sources", str(docs)])
            order = [c[0] for c in CALLS if c[0].startswith("fetch.")]
            check("fetch runs the kinds in FETCH_WHAT order, with transport None (fetch's default), then writes the "
                  "manifest; refetch defaults to the config's card_image_counts sources",
                  rc == 0 and order == ["fetch.taxonomy", "fetch.known_items", "fetch.fetch_cards", "fetch.fetch_kt7",
                                        "fetch.refetch", "fetch.refetch", "fetch.write_manifest"]
                  and one("fetch.taxonomy")[2] == {"domain": DOM_OBJ[0], "names_from": names,
                                                   "out_path": out / "taxonomy_cache.json", "transport": None}
                  and one("fetch.known_items")[2] == {"sources": [docs], "out_path": out / "known_items_v1.json"}
                  and [c[2]["slug"] for c in calls("fetch.refetch")] == ["s_a", "s_b"]
                  and one("fetch.write_manifest")[2] == {"out_dir": out, "prereg": PRE_OBJ[0], "testing": False},
                  (rc, se, order))
            check("a function with a prereg keyword the call does not bind gets the CLI's loaded prereg (context), "
                  "and --testing reaches it",
                  run(["fetch", "--prereg", pre, "--out", str(out), "--what", "cards", "--testing"])[0] == 0
                  and calls("fetch.write_manifest")[-1][2]["prereg"] is PRE_OBJ[0]
                  and calls("fetch.write_manifest")[-1][2]["testing"] is True)
            CALLS[:] = []
            run(["fetch", "--prereg", pre, "--out", str(out), "--what", "refetch", "--slug", "s_z"])
            check("fetch --slug names the refetch sources", [c[2]["slug"] for c in calls("fetch.refetch")] == ["s_z"])
            check("fetch refuses: taxonomy without --names-from, --names-from without taxonomy, known-items without "
                  "--sources, --slug without refetch, an unknown or repeated kind",
                  all(run(["fetch", "--prereg", pre, "--out", str(out)] + b)[0] == 2 for b in (
                      ["--what", "taxonomy"], ["--what", "cards", "--names-from", str(names)],
                      ["--what", "known-items"], ["--what", "cards", "--slug", "s"], ["--what", "stars"],
                      ["--what", "cards,cards"])))

            # exit codes
            out = fresh_out("codes")
            RAISE["draw.draw"] = F.DrawError("frame G1 below its minimum")
            rc, so, se = run(["draw", "--prereg", pre, "--out", str(out)])
            check("a FunnelError subclass from the engine exits 2 with its message on stderr",
                  rc == 2 and "frame G1 below its minimum" in se and "Traceback" not in se, se)
            RAISE["draw.draw"] = ValueError("a bug")
            rc, so, se = run(["draw", "--prereg", pre, "--out", str(out)])
            check("any other exception exits 1 with its traceback", rc == 1 and "Traceback" in se and "a bug" in se)
            RAISE.clear()
            rc, so, se = run(["draw", "--prereg", str(TMP / "missing_prereg.json"), "--out", str(out)])
            check("a prereg the loader refuses exits 2", rc == 2 and "no prereg" in se, se)
            try:
                M._module("no_such_engine_module")
                ok = False
            except F.CLIError as e:
                ok = "not installed" in str(e)
            check("an engine module that is not there is a refusal (CLIError), not a crash", ok)
    finally:
        M.cuda_available = saved_cuda


# ------------------------------------------------------ estimate reruns
def test_estimate_rerun():
    """The real estimate.evaluate writes audit_v1.json and audit_v1.md itself, with a fresh built_utc, and raises
    EstimateError after writing an invalid audit. A fake with that behaviour: a rerun that reproduces the audit
    must leave the files' bytes (and so every recorded sha256 of them) as they were."""
    print("estimate reruns with an evaluator that writes its own files")
    state = {"n": 0, "valid": True, "value": 1}

    def evaluate(prereg_path, funnel_dir, adapter, domain=None, testing=False, write=True):
        state["n"] += 1
        audit = {"format": "funnel-audit/1", "built_utc": "2026-09-28T00:00:%02dZ" % state["n"],
                 "valid": state["valid"], "hypotheses": {"H1": {"estimate": state["value"]}},
                 "inputs": {"x": {"path": "p", "sha256": "s", "built_utc": "t%d" % state["n"]}}}
        if write:
            fd = pathlib.Path(funnel_dir)
            (fd / "audit_v1.json").write_text(json.dumps(audit, sort_keys=True, indent=1) + "\n")
            (fd / "audit_v1.md").write_text("# audit built %s\n" % audit["built_utc"])
        if not audit["valid"]:
            raise F.EstimateError("calibration overlap: written with valid=false")
        return audit

    saved_cuda = M.cuda_available
    M.cuda_available = lambda: False
    try:
        with Fakes() as fk:
            fk.mods["estimate"].evaluate = evaluate
            fk.mods["estimate"].render_md = lambda a: "# audit built %s\n" % a["built_utc"]
            out = fresh_out("estimate_rerun")
            pre = str(PREREG_COPY)
            rc1, _o, e1 = run(["estimate", "--prereg", pre, "--out", str(out)])
            first = ((out / "audit_v1.json").read_bytes(), (out / "audit_v1.md").read_bytes())
            rc2, _o, e2 = run(["estimate", "--prereg", pre, "--out", str(out)])
            second = ((out / "audit_v1.json").read_bytes(), (out / "audit_v1.md").read_bytes())
            check("a rerun that reproduces the audit (volatile keys aside) keeps audit_v1.json's and audit_v1.md's "
                  "bytes, although the evaluator rewrote them",
                  rc1 == 0 and rc2 == 0 and state["n"] == 2 and first == second
                  and b"00:00:01Z" in second[0] and not list(out.glob("*.tmp")), (rc1, rc2, e1, e2))
            state["value"] = 2
            rc3, _o, _e = run(["estimate", "--prereg", pre, "--out", str(out)])
            third = (out / "audit_v1.json").read_bytes()
            check("an audit that changed replaces the file",
                  rc3 == 0 and third != first[0] and json.loads(third)["hypotheses"]["H1"]["estimate"] == 2)
            state["valid"] = False
            rc4, _o, e4 = run(["estimate", "--prereg", pre, "--out", str(out)])
            fourth = json.loads((out / "audit_v1.json").read_text())
            check("an invalid audit exits 2 and stays written with valid false (nothing may cite it)",
                  rc4 == 2 and "calibration overlap" in e4 and fourth["valid"] is False, e4)
            before = (out / "audit_v1.json").read_bytes()
            rc5, _o, _e = run(["estimate", "--prereg", pre, "--out", str(out)])
            check("... and a rerun of the same invalid audit keeps its bytes too",
                  rc5 == 2 and (out / "audit_v1.json").read_bytes() == before)
    finally:
        M.cuda_available = saved_cuda


# ------------------------------------------------ real engine signatures
def test_real_signatures():
    """Every verb against the REAL engine modules and adapter, each function replaced by a recorder that keeps
    its real signature and binds the call to it: the CLI's calls match the functions it will meet on the
    cluster, not only the fakes above."""
    print("every verb's call binds to the real engine function's signature")
    import importlib
    import inspect
    try:
        D = importlib.import_module(PKG + ".domain")
        pre = D.load_prereg(PREREG_COPY)
        dom = D.load(pre.domain_name)
        mods = {n: importlib.import_module("%s.%s" % (PKG, n)) for n in (
            "adapters", "leak", "embed", "judges", "qualify", "draw", "sheets", "rl", "estimate", "relation",
            "recover", "fetch")}
        ad = mods["adapters"].load(dom.adapter)
    except Exception as e:                    # noqa: BLE001 - an optional dependency or a module not there yet
        SKIPS.append("real engine signatures")
        print("  SKIP the real engine modules do not import here: %s" % e)
        return
    bound = []

    class Emb(object):
        name = "fake"
        dim = 4

    class Client(object):
        def __init__(self, endpoint, model, transport=None, num_ctx=8192):
            self.endpoint, self.model = endpoint, model

        def check_vision(self):
            bound.append(("rl.OllamaClient.check_vision", True))

    def recorder(owner, attr, ret):
        real = getattr(owner, attr)
        sig = inspect.signature(real)
        label = "%s.%s" % (getattr(owner, "__name__", owner).rsplit(".", 1)[-1], attr)

        def fn(*a, **kw):
            try:
                sig.bind(*a, **kw)
                ok = True
            except TypeError as e:
                ok = "%s" % e
            bound.append((label, ok))
            return ret
        fn.__signature__ = sig
        fn.__name__ = attr
        fn.__module__ = getattr(real, "__module__", None)
        return fn

    patches = [(ad, "census", {"format": "funnel-census/1"}), (ad, "ledger_from_summaries", {"format": "x"}),
               (ad, "crop_table", "CROPS"), (mods["leak"], "run", {}), (mods["leak"], "run_v2", {}),
               (mods["embed"], "embed_crops", {}),
               (mods["embed"], "embed_table", {}), (mods["judges"], "score_all", {}), (mods["qualify"], "judges", {}),
               (mods["qualify"], "rl", {}), (mods["draw"], "draw", {}), (mods["sheets"], "run", {}),
               (mods["rl"], "run_rl_b", {}), (mods["rl"], "ingest", {}),
               (mods["estimate"], "evaluate", {"format": "funnel-audit/1", "valid": True}),
               (mods["relation"], "run_geometry", {}), (mods["relation"], "run_relation", {}),
               (mods["recover"], "run", {}), (mods["recover"], "arms", {}), (mods["fetch"], "taxonomy", {}),
               (mods["fetch"], "known_items", {}), (mods["fetch"], "fetch_cards", {}), (mods["fetch"], "fetch_kt7", {}),
               (mods["fetch"], "refetch", {}), (mods["fetch"], "write_manifest", {})]
    tag, _factory = M.step1_embedder(ad)
    saved = [(o, a, getattr(o, a)) for o, a, _r in patches]
    saved += [(ad, tag + "_embedder", getattr(ad, tag + "_embedder")),
              (mods["embed"], "Dinov2Embedder", mods["embed"].Dinov2Embedder),
              (mods["rl"], "OllamaClient", mods["rl"].OllamaClient),
              (mods["estimate"], "render_md", mods["estimate"].render_md)]
    saved_cuda = M.cuda_available
    try:
        for o, a, r in patches:
            setattr(o, a, recorder(o, a, r))
        setattr(ad, tag + "_embedder", lambda: Emb())
        mods["embed"].Dinov2Embedder = lambda model, pooling: Emb()
        mods["rl"].OllamaClient = Client
        mods["estimate"].render_md = lambda audit: "# audit\n"
        M.cuda_available = lambda: True
        p = str(PREREG_COPY)
        out = fresh_out("real_sig")
        (out / "taxonomy_cache.json").write_text("{}")
        (out / "known_items_v1.json").write_text("{}")
        (out / "census_v0.json").write_text("[]")
        (out / "refetch").mkdir()
        (out / "refetch" / "crops_refetch.csv").write_text("x\n")
        for d in ("sheets_v1", "rl_answers/RL-A"):
            (out / d).mkdir(parents=True)
        (out / "audit_v1.json").write_text("{}")
        (out / "class_maps.json").write_text("{}")
        exp = pathlib.Path(os.environ["INC_DIR"]) / "realloop_sig"
        exp.mkdir(parents=True, exist_ok=True)
        (exp / "exp.json").write_text("{}")
        base = TMP / "base_sig.jsonl"
        base.write_text("")
        names = TMP / "names_sig.json"
        names.write_text("{}")
        common = ["--prereg", p, "--out", str(out), "--force"]
        runs = [
            ["census"] + common,
            ["census", "--summaries-only", "--prereg", p, "--out", str(out)],
            ["leak"] + common,
            ["embed-judges", "--stage", "all", "--nshards", "2"] + common,
            ["embed-judges", "--stage", "embed", "--refetch"] + common,
            ["qualify"] + common, ["qualify", "--rl"] + common, ["draw"] + common, ["sheets"] + common,
            ["rl-b", "--endpoint", "http://127.0.0.1:1", "--prereg", p, "--out", str(out)],
            ["ingest", "--prereg", p, "--out", str(out)],
            ["estimate", "--prereg", p, "--out", str(out)],
            ["map", "--part", "geometry"] + common, ["map", "--part", "relation", "--prereg", p, "--out", str(out)],
            ["recover", "--audit", str(out / "audit_v1.json"), "--maps", str(out / "class_maps.json"), "--policy",
             "R-A,R-V"] + common,
            ["recover", "--arms", "--realloop", str(exp), "--base", str(base), "--prereg", p, "--out", str(out)],
            ["fetch", "--what", "taxonomy,known-items,cards,kt7,refetch", "--names-from", str(names), "--sources",
             str(TMP), "--prereg", p, "--out", str(out), "--testing"],
        ]
        codes = []
        for argv in runs:
            rc, so, se = run(argv)
            codes.append((argv[0], rc, se.strip().splitlines()[-1:] if rc else ""))
        bad = [b for b in bound if b[1] is not True]
        called = {b[0] for b in bound}
        # the real prereg's amendment A2 requires copy detector version 2: leak calls leak.run_v2
        want = {"inc_step1.census", "inc_step1.ledger_from_summaries", "inc_step1.crop_table", "leak.run_v2",
                "embed.embed_crops", "embed.embed_table", "judges.score_all", "qualify.judges", "qualify.rl",
                "draw.draw", "sheets.run", "rl.run_rl_b", "rl.ingest", "estimate.evaluate", "relation.run_geometry",
                "relation.run_relation", "recover.run", "recover.arms", "fetch.taxonomy", "fetch.known_items",
                "fetch.fetch_cards", "fetch.fetch_kt7", "fetch.refetch", "fetch.write_manifest",
                "rl.OllamaClient.check_vision"}
        check("every verb exits 0 and every call it makes binds to the real function's signature (%d calls; "
              "--force reaches only functions that take it)" % len(bound),
              all(rc == 0 for _v, rc, _e in codes) and not bad and want <= called,
              ([c for c in codes if c[1]], bad[:5], sorted(want - called)))
    finally:
        M.cuda_available = saved_cuda
        for o, a, v in saved:
            setattr(o, a, v)


# --------------------------------------------------------- real domain loader
def test_real_domain():
    print("the real domain loader")
    import importlib
    try:
        D = importlib.import_module(PKG + ".domain")
    except ImportError as e:
        SKIPS.append("real domain loader")
        print("  SKIP the domain module does not import: %s" % e)
        return
    pre = D.load_prereg(PREREG_COPY)
    cfg = pathlib.Path(F.__file__).resolve().parent / "domains" / ("%s.json" % pre.domain_name)
    if not cfg.is_file():
        SKIPS.append("real domain config")
        print("  SKIP %s is not there yet (G-data)" % cfg)
        return
    saved = {n: sys.modules.get("%s.%s" % (PKG, n)) for n in ("sheets", "adapters")}
    got = {}

    def run_fn(prereg, domain, funnel_dir, adapter):
        got.update(prereg=prereg, domain=domain, adapter=adapter)
        return {"format": "funnel-sheets/1"}
    sys.modules[PKG + ".sheets"] = types.SimpleNamespace(run=run_fn)
    sys.modules[PKG + ".adapters"] = types.SimpleNamespace(load=lambda name: ("adapter", name), INTERFACE=())
    try:
        out = fresh_out("real")
        rc, so, se = run(["sheets", "--prereg", str(PREREG_COPY), "--out", str(out)])
        check("with the real loader, a verb gets the Prereg (the committed prereg's core sha256) and the Domain the "
              "prereg names, and the adapter the config names",
              rc == 0 and isinstance(got.get("prereg"), D.Prereg)
              and got["prereg"].core_sha256 == D.prereg_core_sha256(json.loads(PREREG.read_text()))
              and isinstance(got.get("domain"), D.Domain) and got["domain"].name == pre.domain_name
              and got["adapter"] == ("adapter", got["domain"].adapter), (rc, se))
        edited = TMP / "repo" / "docs" / "FUNNEL_AUDIT.md"
        text = edited.read_bytes()
        edited.write_bytes(text + b"\nedited\n")
        try:
            rc, so, se = run(["sheets", "--prereg", str(PREREG_COPY), "--out", str(out)])
            check("an edited contract is refused through the prereg loader (exit 2)", rc == 2 and "contract" in se, se)
        finally:
            edited.write_bytes(text)
    finally:
        for n, m in saved.items():
            if m is None:
                sys.modules.pop("%s.%s" % (PKG, n), None)
            else:
                sys.modules["%s.%s" % (PKG, n)] = m

    # the real adapter: census --summaries-only on copies of the local Step 1 summaries and census_v0 (F2a)
    real = ROOT / "results" / "framework" / "inc"
    need = [real / "step1" / n for n in ("pool_summary.json", "admit_summary.json", "select_summary.json",
                                         "calibration.json")] + [real / "funnel" / "census_v0.json"]
    missing = [str(p) for p in need if not p.is_file()]
    try:
        ad = importlib.import_module(PKG + ".adapters").load(D.load(pre.domain_name).adapter)
        has = callable(getattr(ad, "ledger_from_summaries", None))
    except Exception as e:                    # noqa: BLE001 - reported as a skip with its reason
        has = False
        missing.append("adapter: %s" % e)
    if missing or not has:
        SKIPS.append("real census --summaries-only")
        print("  SKIP real census --summaries-only: %s" % (missing or "no ledger_from_summaries"))
        return
    step1 = F.STEP1_DIR
    step1.mkdir(parents=True, exist_ok=True)
    for p in need[:4]:
        shutil.copyfile(p, step1 / p.name)
    out = fresh_out("real_census")
    shutil.copyfile(need[4], out / "census_v0.json")
    rc, so, se = run(["census", "--summaries-only", "--prereg", str(PREREG_COPY), "--out", str(out)])
    led = json.loads((out / "funnel_ledger.json").read_text()) if (out / "funnel_ledger.json").is_file() else {}
    admit = json.loads(need[1].read_text())
    kept = {s.get("role"): s.get("kept") for s in led.get("stages", [])}
    check("census --summaries-only through the CLI and the real adapter writes funnel_ledger.json "
          "(funnel-ledger/1, derivation summaries) whose target check keeps admit_summary.json's %d verified boxes "
          "and whose other check keeps its %d other_ok boxes" % (admit["boxes"]["verified"], admit["boxes"]["other_ok"]),
          rc == 0 and led.get("format") == "funnel-ledger/1" and led.get("derivation") == "summaries"
          and kept.get("target_check") == admit["boxes"]["verified"]
          and kept.get("other_check") == admit["boxes"]["other_ok"]
          and last_line(so).startswith("[funnel] census: "), (rc, se[-500:], kept))


# ------------------------------------------------------------ domain-free
FIXED_TERMS = ("cwd12", "otherplant", "bioclip", "ndsu", "cottonweed")


def test_domain_free():
    print("domain-free")
    text = pathlib.Path(M.__file__).read_text().lower()
    for own in ("weed_optimizer_framework", "weed_llm_benchmark"):
        text = text.replace(own, "")
    subs, toks = set(FIXED_TERMS), set()
    try:
        import importlib
        D = importlib.import_module(PKG + ".domain")
        for cfg in sorted((pathlib.Path(F.__file__).resolve().parent / "domains").glob("*.json")):
            t = D.load(cfg).terms()
            subs |= {s.lower() for s in t["substring"]}
            toks |= {s.lower() for s in t["token"]}
    except Exception as e:                    # noqa: BLE001 - the fixed list still runs
        print("       (domain terms not loaded: %s)" % e)
    words = set(re.findall(r"[a-z0-9]+", text))
    found = sorted([s for s in subs if s and s in text] + [t for t in toks if t in words])
    check("funnel/__main__.py names no domain term (%d substring and %d token terms)" % (len(subs), len(toks)),
          not found, found)


def main():
    test_tables()
    test_parser()
    test_dispatch()
    test_estimate_rerun()
    test_real_signatures()
    test_real_domain()
    test_domain_free()


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), SKIPS))
    sys.exit(1 if FAILURES else 0)
