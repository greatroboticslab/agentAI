"""INC Step 1: which harvested sources hold plant photographs at all.

    python -m weed_optimizer_framework.tools.inc.relevance build [--sample 300] [--seed 0]
        [--base-dir INC_DIR/step1] [--out BASE_DIR/relevance.json] [--force]

Contract: docs/INCREMENTAL_PROTOCOL.md, Step 1 ("Source relevance") and Steps 2-3.

Why. The Step 1 verifier (inc/verify.py) certifies species labels, not
relevance. A box labelled OtherPlant is admitted unless the probe confidently
calls it a cwd12 species, so an OtherPlant box around a video-game player or a
car is admitted as readily as one around a weed. The increment pool that
inc/select.py leaves (increment_pool.jsonl) is almost all OtherPlant boxes, and
it holds whole sources that are not plant photographs. This module measures,
per source, whether its boxes look like plants to BioCLIP-2 itself, against a
threshold calibrated on train_core, and records which sources pass. No source
is named by hand: select increments and realloop build read pass / fail from
this file.

How.
  * Zero-shot BioCLIP-2. The text tower of the model that embedded the crops
    (open_clip, the embedder named in the embedding shards) encodes PROMPTS.
    Text and image features are L2-normalised; a logit is the model's own
    logit scale times the cosine. A crop's P(plant) is the softmax over the
    decision prompts (PROMPTS["plant"] + PROMPTS["non_plant"]), summed over
    the plant prompts. The tokenizer must be the one the model's own open_clip
    config names, and the snapshot commit and weights-file sha256 are
    recorded (BioclipTextEncoder).
  * Image features are the Step 1 crop embeddings (step1/emb, read through
    verify.load_embeddings). No image is opened and nothing is embedded again.
  * Calibration. Every train_core crop with a feature (all genuine weeds):
    tau = the CAL_PERCENTILE-th (5th) percentile of their P(plant), so 95 %
    of train_core crops score at or above it. Weeds from other cameras or
    domains can score lower (the false-negative risk; see the name check
    below). The distribution is recorded.
  * Calibration check (fail closed). tau must be >= TAU_MIN (0.5): at least
    95 % of train_core crops are judged more plant than non-plant. A lower tau
    means the prompts do not separate genuine weeds from the non-plant set
    (for example, seedling crops read as 'bare soil'); tau would then fall
    towards 0 and pass nearly every source. Such a file is still written, for
    a person to read, but the build exits with an error and load() refuses it.
  * Per source of increment_pool.jsonl and, separately, of base_selected.jsonl
    (each table on that manifest's own images): a seeded sample of up to
    --sample crops with a feature, numpy default_rng(stable_int(SEED_TEXT %
    (seed, table, source))) without replacement over the source's usable crop
    ids in ascending order. The source passes when the median P(plant) of its
    sample is >= tau. A source with fewer than MIN_CROPS usable crops is
    'insufficient' and does not pass (a source cannot be cleared on no
    evidence).
  * Reported only: PROMPTS["leaf_disease"], a third set kept out of the
    softmax so that it cannot move P(plant). Per source and on train_core,
    the share of crops whose top prompt over all prompts is one of them.
    It never changes pass / fail.
  * Reported only: the name check. The non-passing sources whose name holds
    a word of PLANT_NAME_WORDS are listed for a person, as possible false
    negatives (plant imagery the rule misjudged). The word list never
    changes a status and names no source.
  * base_selected's sources are reported for the record; base B is not
    rebuilt here.

Inputs: the inc/select.py build in --base-dir (select_summary.json;
increment_pool.jsonl and base_selected.jsonl must hash as it recorded), and
the crops.csv and embedding shards that build read (the same sha256 for
crops.csv and every shard file, the same shard set).

Outputs: OUT (default <base-dir>/relevance.json) and OUT.md beside it, both
written atomically. Deterministic: the same inputs and parameters give the
same numbers. A rerun with the same inputs and parameters is a no-op (and
exits with the same error when the calibration check failed); a rerun that
would overwrite a file made from other inputs or parameters refuses unless
--force, since an increment draw may have recorded its sha256.

Consumers: select.increments(..., relevance=PATH) and realloop build
(--relevance, default <base-dir>/relevance.json) read it through load(). It
refuses a file made for another increment pool, a file whose calibration
check failed, and a file in which a status (or the check) does not follow
from its recorded numbers under the rule. That is a consistency check, not a
seal: the file's sha256, recorded by every draw, identifies the exact file
used.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

from . import common as C
from . import select as S
from . import verify as V

# The zero-shot prompt sets, in one place; the output records them.
#   plant, non_plant  the decision prompts: P(plant) is the softmax over both
#                     sets, summed over the plant set;
#   leaf_disease      reported only: kept out of the softmax (it cannot change
#                     P(plant) or pass / fail); the share of crops whose top
#                     prompt over every prompt is one of these.
PROMPTS = {
    "plant": ("a photo of a plant", "a photo of a weed", "a photo of a crop plant",
              "a photo of a seedling", "Plantae"),
    "non_plant": ("a photo of a person", "a screenshot of a video game", "a photo of a vehicle",
                  "a photo of an animal", "a photo of an insect", "a photo of bare soil",
                  "a photo of a building", "text"),
    "leaf_disease": ("a close-up photo of a diseased leaf", "a photo of a leaf with disease spots",
                     "a close-up photo of a single leaf"),
}
PLANT, NON_PLANT, DIAGNOSTIC = "plant", "non_plant", "leaf_disease"
SETS = (PLANT, NON_PLANT, DIAGNOSTIC)
DECISION_SETS = (PLANT, NON_PLANT)
SAMPLE = 300                  # crops per source (at most)
CAL_PERCENTILE = 5.0          # tau: this percentile of train_core's P(plant)
TAU_MIN = 0.5                 # the calibration check: tau below it fails closed
MIN_CROPS = 20                # usable crops a source needs to be judged
# Reported only (the name check): a non-passing source whose name holds one of
# these words may be plant imagery the zero-shot rule misjudged (another
# camera, altitude or domain than train_core's). The list puts such sources in
# front of a person; it never changes a status and names no source.
PLANT_NAME_WORDS = ("weed", "crop", "plant", "seed", "leaf", "agri", "farm", "field")
HF_HUB = "hf-hub:"
TABLES = ("increment_pool", "base_selected")
TABLE_FILE = {"increment_pool": S.POOL, "base_selected": S.BASE_SELECTED}
OUT_NAME = "relevance.json"
FORMAT = "inc.relevance/1"
PASS, FAIL, INSUFFICIENT = "pass", "fail", "insufficient"
STATUSES = (PASS, FAIL, INSUFFICIENT)
SEED_TEXT = "inc.relevance/%d/%s/%s"          # (seed, table, source)
CHUNK = 65_536
DIST_PCTS = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
HIST_BINS = 20
WHAT = "INC Step 1 source relevance: zero-shot BioCLIP-2 P(plant) per source (inc/relevance.py)"
RULE = ("tau = the %gth percentile of P(plant) over every train_core crop with a feature; a source "
        "passes when the median P(plant) of its sample (up to --sample crops with a feature, seeded) is "
        ">= tau; a source with fewer than %d crops with a feature is insufficient and does not pass"
        % (CAL_PERCENTILE, MIN_CROPS))
CHECK_RULE = ("the calibration is usable only when tau >= %g, that is, when at least %g %% of train_core crops "
              "are judged more plant than non-plant (P(plant) >= 0.5); otherwise the prompts do not separate "
              "genuine weeds from the non-plant set, the file is written for a person to read, and "
              "relevance.load refuses it" % (TAU_MIN, 100.0 - CAL_PERCENTILE))


class RelevanceError(RuntimeError):
    """A condition under which the relevance file must not be written or used."""


def log(msg):
    print("[inc.relevance %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _r(x, digits=6):
    return None if x is None else round(float(x), digits)


# ------------------------------------------------------------ text encoder
def hub_snapshot(path):
    """The Hugging Face cache snapshot (commit) a resolved hub file lives in:
    .../snapshots/<40-hex commit>/<file>, else None."""
    p = Path(str(path))
    if p.parent.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", p.parent.name):
        return p.parent.name
    return None


def check_tokenizer(tokenizer, text_cfg, model_ctx, model_vocab, simple_cls, hf_cls):
    """The tokenizer open_clip built must be the one the model's own config
    names. open_clip.get_tokenizer falls back to HFTokenizer(<repo id>) when it
    cannot read an hf-hub model's config; for BioCLIP-2 that fallback loads
    (the repo ships tokenizer files) and emits ids that fit the model, so a
    check on the tokens alone cannot see it. text_cfg without an
    hf_tokenizer_name means open_clip's SimpleTokenizer; with one, HFTokenizer.
    The context length (and a SimpleTokenizer's vocabulary) must be the
    model's. Returns a record of the tokenizer."""
    hf_name = text_cfg.get("hf_tokenizer_name") or None
    want = hf_cls if hf_name else simple_cls
    ctx = getattr(tokenizer, "context_length", None)
    want_ctx = [int(c) for c in (text_cfg.get("context_length"), model_ctx) if c is not None]
    bad = []
    if not isinstance(tokenizer, want):
        bad.append("it is a %s, the config names %s" % (type(tokenizer).__name__, want.__name__))
    if ctx is None or not want_ctx or any(int(ctx) != c for c in want_ctx):
        bad.append("context length %s; config %s, model %s" % (ctx, text_cfg.get("context_length"), model_ctx))
    voc = getattr(tokenizer, "vocab_size", None)
    if want is simple_cls and model_vocab is not None and voc is not None and int(voc) != int(model_vocab):
        bad.append("vocabulary %s; model %s" % (voc, model_vocab))
    if bad:
        raise RelevanceError("the tokenizer is not the model's own: %s" % "; ".join(bad))
    return {"class": type(tokenizer).__name__, "hf_tokenizer_name": hf_name,
            "context_length": int(ctx), "vocab_size": None if voc is None else int(voc)}


class BioclipTextEncoder:
    """The text tower of the model that embedded the crops (open_clip; the
    same loader as verify's BioclipEmbedder, semisup_labeler._load_backbone).
    Called with prompts, returns their raw text features [n, dim].

    Pinned to the model's own files, from the Hugging Face cache (offline on
    the cluster), through open_clip's own resolver (download_pretrained_from_hf,
    the call create_model makes, safetensors first):
      * the tokenizer must be the one the model's open_clip config names
        (check_tokenizer), and every call's tokens must fit the model;
      * the weights file must resolve: open_clip builds a randomly initialised
        model, with only a logged warning, when it cannot find the weights;
      * the snapshot commit and the sha256 of the config and weights files
        are recorded (self.provenance)."""

    def __init__(self, model_name=V.EMBEDDER_NAME):
        import open_clip
        from open_clip.pretrained import download_pretrained_from_hf
        from open_clip.tokenizer import HFTokenizer, SimpleTokenizer
        from .. import semisup_labeler as SL
        if not str(model_name).startswith(HF_HUB):
            raise RelevanceError("%s: only %s models are supported (their files are pinned from the hub cache)"
                                 % (model_name, HF_HUB))
        repo = model_name[len(HF_HUB):]
        self.model = model_name
        try:
            cfg_path = Path(download_pretrained_from_hf(repo, filename="open_clip_config.json"))
            weights = Path(download_pretrained_from_hf(repo))
            with open(cfg_path) as fh:
                cfg = json.load(fh)
        except Exception as e:  # noqa: BLE001 - any failure to resolve the model's own files refuses
            raise RelevanceError("%s: the open_clip config or weights do not resolve (%s: %s)"
                                 % (model_name, type(e).__name__, e))
        snap = hub_snapshot(cfg_path)
        if snap is None or hub_snapshot(weights) != snap:
            raise RelevanceError("%s: the config (%s) and weights (%s) are not files of one hub snapshot"
                                 % (model_name, cfg_path, weights))
        text_cfg = (cfg.get("model_cfg") or {}).get("text_cfg") or {}
        self.device = SL._device()
        log("loading %s on %s (snapshot %s, weights %s)" % (model_name, self.device, snap, weights.name))
        self._model, _pre = SL._load_backbone(model_name, self.device)
        self._tokenizer = open_clip.get_tokenizer(model_name)
        tok = check_tokenizer(self._tokenizer, text_cfg, getattr(self._model, "context_length", None),
                              getattr(self._model, "vocab_size", None), SimpleTokenizer, HFTokenizer)
        self.logit_scale = float(self._model.logit_scale.exp().item())
        bias = getattr(self._model, "logit_bias", None)
        self.logit_bias = None if bias is None else float(bias.item())
        self.name = "%s text tower (open_clip %s, %s)" % (model_name, open_clip.__version__,
                                                         type(self._tokenizer).__name__)
        log("hashing %s" % weights)
        self.provenance = {
            "repo": repo, "snapshot": snap, "open_clip": open_clip.__version__,
            "config": {"file": cfg_path.name, "sha256": C.sha256_file(cfg_path)},
            "weights": {"file": weights.name, "sha256": C.sha256_file(weights),
                        "bytes": int(os.path.getsize(weights))},
            "tokenizer": tok,
            "note": ("the files open_clip resolves now; the embed stage's shard meta names the model, "
                     "not its snapshot, so the two are not compared")}

    def __call__(self, prompts):
        import torch
        tokens = self._tokenizer(list(prompts))
        ctx = getattr(self._model, "context_length", None)
        vocab = getattr(self._model, "vocab_size", None)
        if ((ctx is not None and tokens.shape[-1] != int(ctx))
                or (vocab is not None and int(tokens.max()) >= int(vocab))):
            raise RelevanceError("the tokenizer (%s) does not fit %s: tokens %s, max id %d; model context %s, "
                                 "vocabulary %s" % (type(self._tokenizer).__name__, self.model,
                                                    tuple(tokens.shape), int(tokens.max()), ctx, vocab))
        with torch.no_grad():
            return self._model.encode_text(tokens.to(self.device)).float().cpu().numpy()


def prompt_table():
    """(prompts, the set of each prompt), every set in SETS order."""
    prompts, sets = [], []
    for s in SETS:
        for p in PROMPTS[s]:
            prompts.append(p)
            sets.append(s)
    return prompts, sets


def text_features(encoder, prompts, dim):
    """L2-normalised text features [len(prompts), dim] (float64)."""
    T = np.asarray(encoder(list(prompts)), dtype=np.float64)
    if T.shape != (len(prompts), int(dim)):
        raise RelevanceError("the text encoder returned %s for %d prompts; the crop features have dim %d"
                             % (T.shape, len(prompts), dim))
    n = np.linalg.norm(T, axis=1, keepdims=True)
    if not np.isfinite(T).all() or (n <= 0).any():
        raise RelevanceError("the text encoder returned a non-finite or zero feature")
    return T / n


def scores(X, idx, T, scale, sets):
    """(P(plant), top prompt) of the crops idx of X: P(plant) = softmax of
    scale * cosine over the decision prompts, summed over the plant prompts;
    top = the argmax over every prompt (the diagnostic set included)."""
    sets = np.asarray(sets)
    dec = np.flatnonzero(np.isin(sets, DECISION_SETS))
    plant = np.flatnonzero(sets[dec] == PLANT)
    idx = np.asarray(idx, dtype=np.int64)
    p = np.zeros(len(idx), dtype=np.float64)
    top = np.zeros(len(idx), dtype=np.int64)
    for s in range(0, len(idx), CHUNK):
        Xn = np.asarray(X[idx[s:s + CHUNK]], dtype=np.float64)
        Xn /= np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12)
        L = float(scale) * (Xn @ T.T)
        top[s:s + CHUNK] = np.argmax(L, axis=1)
        Ld = L[:, dec]
        E = np.exp(Ld - Ld.max(axis=1, keepdims=True))
        p[s:s + CHUNK] = E[:, plant].sum(axis=1) / E.sum(axis=1)
    return p, top


def status_of(usable, median, tau):
    """The rule: insufficient below MIN_CROPS usable crops, else pass iff the
    median P(plant) is >= tau."""
    if int(usable) < MIN_CROPS or median is None:
        return INSUFFICIENT
    return PASS if float(median) >= float(tau) else FAIL


def _top_block(top, sets, prompts):
    n = len(top)
    sets = np.asarray(sets)
    share = {s: _r(np.mean(sets[top] == s)) if n else None for s in SETS}
    counts = collections.Counter(prompts[int(t)] for t in top)
    return share, {p: int(counts[p]) for p in prompts if counts.get(p)}


def check_of(tau):
    """The calibration check, from tau alone (load() re-derives it)."""
    return float(tau) >= TAU_MIN


def plant_name_words(source):
    """The PLANT_NAME_WORDS in a source's name (the reported-only name check)."""
    s = str(source).lower()
    return [w for w in PLANT_NAME_WORDS if w in s]


# ------------------------------------------------------------ the numbers
# P(plant) statistics are stored unrounded: at a logit scale near 100 they pile
# up next to 0 and 1, where rounding would merge a passing and a failing value.
def calibrate(crops, X, T, scale, sets, prompts):
    """tau, the calibration check and the distribution of P(plant) over
    train_core's crops."""
    core = crops.where("core").astype(np.int64)
    fin = S._finite_rows(X, core) if len(core) else np.zeros(0, dtype=bool)
    use = core[fin]
    if not len(use):
        raise RelevanceError("crops.csv holds no train_core crop with a feature; nothing to calibrate on")
    p, top = scores(X, use, T, scale, sets)
    tau = float(np.percentile(p, CAL_PERCENTILE))
    hist, edges = np.histogram(p, bins=HIST_BINS, range=(0.0, 1.0))
    share, per_prompt = _top_block(top, sets, prompts)
    return tau, {
        "set": "every train_core crop with a feature (crops.csv set 'core'); all genuine weeds",
        "crops": int(len(use)), "crops_without_feature": int(len(core) - len(use)),
        "percentile": CAL_PERCENTILE, "tau": tau,
        "check": {"tau_min": TAU_MIN, "ok": check_of(tau),
                  "share_below_0.5": _r(np.mean(p < 0.5)), "rule": CHECK_RULE},
        "share_at_or_above_tau": _r(np.mean(p >= tau)),
        "mean": float(p.mean()),
        "distribution": {"p%d" % q: float(np.percentile(p, q)) for q in DIST_PCTS},
        "histogram": {"edges": [_r(e, 4) for e in edges], "counts": [int(c) for c in hist]},
        "top_set_share": share, "top_prompts": per_prompt}


def source_table(table, rows, crops, X, pool_ids, T, scale, sets, prompts, tau, sample, seed):
    """Per source of rows (one manifest): the seeded sample, its P(plant)
    statistics and status. pool_ids: crops.csv's pool crop ids, ascending."""
    src_of = {r["key"]: r["source"] for r in rows}
    images = collections.Counter(r["source"] for r in rows)
    by_src = collections.defaultdict(list)
    for i in pool_ids:
        s = src_of.get(crops.key[i])
        if s is not None:
            by_src[s].append(int(i))
    out = {}
    for src in sorted(images):
        ids = np.array(by_src.get(src, []), dtype=np.int64)
        usable = ids[S._finite_rows(X, ids)] if len(ids) else ids
        if len(usable) > sample:
            rng = np.random.default_rng(C.stable_int(SEED_TEXT % (seed, table, src)))
            pick = usable[np.sort(rng.choice(len(usable), size=sample, replace=False))]
        else:
            pick = usable
        if len(pick):
            p, top = scores(X, pick, T, scale, sets)
            median = float(np.median(p))
            share, per_prompt = _top_block(top, sets, prompts)
            stats = {"p_plant_mean": float(p.mean()),
                     "p_plant_quartiles": [float(np.percentile(p, 25)), float(np.percentile(p, 75))],
                     "share_at_or_above_tau": _r(np.mean(p >= tau)),
                     "top_set_share": share, "top_prompts": per_prompt}
        else:
            median = None
            stats = {"p_plant_mean": None, "p_plant_quartiles": None, "share_at_or_above_tau": None,
                     "top_set_share": {s: None for s in SETS}, "top_prompts": {}}
        st = status_of(len(usable), median, tau)
        out[src] = dict(stats, images=int(images[src]),
                        images_with_usable_crops=len({crops.key[i] for i in usable}),
                        crops=int(len(ids)), crops_usable=int(len(usable)), sampled=int(len(pick)),
                        p_plant_median=median, status=st, passes=st == PASS)
    by_status = collections.Counter(e["status"] for e in out.values())
    img_status = collections.Counter()
    for e in out.values():
        img_status[e["status"]] += e["images"]
    not_passing = {s: e["images"] for s, e in out.items() if e["status"] != PASS}
    named = {s: {"words": plant_name_words(s), "status": out[s]["status"], "images": out[s]["images"],
                 "p_plant_median": out[s]["p_plant_median"]}
             for s in sorted(not_passing) if plant_name_words(s)}
    return {"images": len(rows), "sources": out,
            "sources_by_status": {s: int(by_status.get(s, 0)) for s in STATUSES},
            "images_by_status": {s: int(img_status.get(s, 0)) for s in STATUSES},
            "not_passing": not_passing, "not_passing_images": int(sum(not_passing.values())),
            "name_check": {"words": list(PLANT_NAME_WORDS), "sources": named,
                           "use": "reported only: non-passing sources whose name holds a plant word, for a person "
                                  "to look at as possible false negatives; never changes a status"}}


# ------------------------------------------------------------------ build
def _select_inputs(base_dir):
    """The select build in base_dir, checked: (summary, inputs record)."""
    sel = S._read_json(base_dir / S.SUMMARY)
    if not isinstance(sel, dict):
        raise RelevanceError("no %s in %s: run `inc.select build` first" % (S.SUMMARY, base_dir))
    if not S._outputs_match(sel, base_dir, (S.BASE_SELECTED, S.POOL)):
        raise RelevanceError("%s or %s in %s changed since the select build that wrote %s"
                             % (S.POOL, S.BASE_SELECTED, base_dir, S.SUMMARY))
    sin = sel.get("inputs") or {}
    emb = (sel.get("crops") or {}).get("embeddings") or {}
    if not (sin.get("crops") or {}).get("sha256") or not emb.get("nshards"):
        raise RelevanceError("%s records no crops.csv or embedding shard set" % (base_dir / S.SUMMARY))
    for f in sin.get("emb_files") or []:
        p = Path(f["path"])
        if not p.is_file() or C.sha256_file(p) != f["sha256"]:
            raise RelevanceError("embedding shard %s is not the one the select build read" % p)
    outs = sel["outputs"]
    inputs = {"select_summary": {"path": str(base_dir / S.SUMMARY),
                                 "sha256": C.sha256_file(base_dir / S.SUMMARY)},
              "crops": {k: sin["crops"].get(k) for k in ("path", "sha256")},
              "emb_files": sin.get("emb_files") or [],
              "embeddings": emb,
              "train_core": {k: (sin.get("train_core") or {}).get(k) for k in ("path", "sha256")}}
    for t in TABLES:
        n = TABLE_FILE[t]
        inputs[t] = {"path": str(base_dir / n), "sha256": outs[n]["sha256"], "rows": outs[n].get("rows")}
    return sel, inputs


def build(sample=SAMPLE, seed=0, base_dir=None, out=None, text_encoder=None, force=False):
    """Write OUT (relevance.json) and OUT.md; returns the record."""
    t0 = time.time()
    if isinstance(sample, bool) or int(sample) != sample or int(sample) < MIN_CROPS:
        raise RelevanceError("--sample must be an integer >= %d (MIN_CROPS), got %r" % (MIN_CROPS, sample))
    if isinstance(seed, bool) or int(seed) != seed or int(seed) < 0:
        raise RelevanceError("--seed must be a non-negative integer, got %r" % (seed,))
    sample, seed = int(sample), int(seed)
    base_dir = Path(os.path.abspath(str(base_dir or S.step1_dir())))
    out = Path(os.path.abspath(str(out or base_dir / OUT_NAME)))
    if out.suffix != ".json":
        raise RelevanceError("--out %s: must end in .json (the .md goes beside it)" % out)
    out_md = out.with_suffix(".md")
    sel, inputs = _select_inputs(base_dir)
    inputs_paths = {Path(inputs[k]["path"]) for k in ("select_summary", "crops") + TABLES}
    if out in inputs_paths or out_md in inputs_paths:
        raise RelevanceError("--out %s would overwrite an input" % out)
    emb_model = str(inputs["embeddings"].get("embedder") or "").split("+")[0]
    prompts, sets = prompt_table()
    params = {"sample": sample, "seed": seed, "calibration_percentile": CAL_PERCENTILE, "tau_min": TAU_MIN,
              "min_crops": MIN_CROPS, "prompts": {s: list(PROMPTS[s]) for s in SETS},
              "decision_sets": list(DECISION_SETS), "diagnostic_set": DIAGNOSTIC,
              "text_model": emb_model, "seed_text": SEED_TEXT, "rule": RULE, "check_rule": CHECK_RULE,
              "plant_name_words": list(PLANT_NAME_WORDS), "versions": {"numpy": np.__version__}}
    old = S._read_json(out) if out.exists() else None
    if old is not None:
        if (old.get("params") == params and old.get("inputs") == inputs and not force
                and out_md.exists()):
            log("up to date: %s (same inputs and parameters)" % out)
            if not ((old.get("calibration") or {}).get("check") or {}).get("ok"):
                raise RelevanceError(_degenerate(old.get("calibration") or {}, out))
            return old
        if not force:
            raise RelevanceError("%s exists from a different build (inputs or parameters differ); an increment "
                                 "draw may have recorded its sha256. Pass --force to overwrite." % out)

    try:
        crops = V.Crops(Path(inputs["crops"]["path"]))
    except V.VerifyError as e:
        raise RelevanceError("crops.csv: %s" % e)
    if crops.sha != inputs["crops"]["sha256"]:
        raise RelevanceError("crops.csv %s is not the one the select build read" % crops.path)
    try:
        X, emb_info = V.load_embeddings(crops, int(inputs["embeddings"]["nshards"]))
    except V.VerifyError as e:
        raise RelevanceError("embeddings: %s" % e)
    if emb_info != inputs["embeddings"]:
        raise RelevanceError("the embedding shards are not the ones the select build read (select %s, now %s)"
                             % (inputs["embeddings"], emb_info))
    log("crops.csv: %d crops; embeddings %s" % (crops.n, emb_info))

    enc = text_encoder if text_encoder is not None else BioclipTextEncoder(emb_model)
    if str(getattr(enc, "model", "")) != emb_model:
        raise RelevanceError("the text encoder is %r's, the crops were embedded by %r: the text tower must be "
                             "the image tower's own model" % (getattr(enc, "model", None), emb_model))
    scale = float(enc.logit_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise RelevanceError("logit scale %r is not a positive number" % scale)
    T = text_features(enc, prompts, emb_info["dim"])
    log("text features: %d prompts (%s), logit scale %.4f"
        % (len(prompts), ", ".join("%s %d" % (s, len(PROMPTS[s])) for s in SETS), scale))

    tau, cal = calibrate(crops, X, T, scale, sets, prompts)
    cal["train_core"] = inputs["train_core"]
    log("calibration: %d train_core crops; tau = p%g of P(plant) = %s (share >= tau %.4f; median %s); "
        "check tau >= %g: %s (share of train_core below 0.5: %.4f)"
        % (cal["crops"], CAL_PERCENTILE, _fmt_p(tau), cal["share_at_or_above_tau"],
           _fmt_p(cal["distribution"]["p50"]), TAU_MIN, "ok" if cal["check"]["ok"] else "FAILED",
           cal["check"]["share_below_0.5"]))
    pool_ids = crops.where("pool").astype(np.int64)
    tables = {}
    for t in TABLES:
        rows = C.read_manifest(inputs[t]["path"])
        blk = source_table(t, rows, crops, X, pool_ids, T, scale, sets, prompts, tau, sample, seed)
        blk.update(manifest=inputs[t]["path"], sha256=inputs[t]["sha256"])
        tables[t] = blk
        log("%s: %d images in %d sources; sources %s; images %s; not passing: %s"
            % (t, blk["images"], len(blk["sources"]), blk["sources_by_status"], blk["images_by_status"],
               ", ".join("%s (%d)" % kv for kv in sorted(blk["not_passing"].items(), key=lambda kv: -kv[1]))
               or "none"))
    tables["increment_pool"]["use"] = ("select increments / realloop build draw the verified and OTHER_HEAVY "
                                       "increments only from passing sources; the UNVERIFIED increment is "
                                       "not filtered")
    tables["base_selected"]["use"] = "for the record only: base B is not rebuilt"
    res = {
        "format": FORMAT, "what": WHAT, "built_utc": _utc(), "seconds": round(time.time() - t0, 1),
        "params": params, "inputs": inputs, "rule": RULE,
        "text_encoder": {"name": str(getattr(enc, "name", enc.model)), "model": enc.model,
                         "logit_scale": scale, "logit_bias": getattr(enc, "logit_bias", None),
                         "dim": int(T.shape[1]), "prompt_order": prompts, "prompt_sets": sets,
                         "features_sha256": hashlib.sha256(np.ascontiguousarray(T).tobytes()).hexdigest(),
                         "provenance": getattr(enc, "provenance", None)},
        "calibration": cal,
        "increment_pool": tables["increment_pool"], "base_selected": tables["base_selected"],
        "outputs": {"json": str(out), "md": str(out_md)},
        "notes": [
            "The Step 1 verifier certifies species labels, not relevance: an OtherPlant box on a non-plant "
            "image is admitted. This file is the relevance filter; no source is named by hand.",
            "P(plant): softmax of logit_scale * cosine(crop, prompt) over the plant and non_plant prompts, "
            "summed over the plant prompts. The leaf_disease prompts are outside the softmax.",
            "top_set_share: the share of the sampled crops whose top prompt over every prompt (leaf_disease "
            "included) belongs to each set. Reported only; it never changes pass / fail.",
            "Each table is computed on its own manifest's images: a source in both tables is judged twice, "
            "on different images. Only increment_pool's status is used.",
            "A crop without a feature (NaN embedding) is not usable; a box too small to crop has no crop row.",
            "tau is calibrated on train_core (cwd12) crops: 95 % of them score at or above it. Weeds from other "
            "cameras or domains can score lower; name_check lists the non-passing sources whose name holds a "
            "plant word, for a person (reported only).",
            "calibration.check: tau must be >= %g, else relevance.load refuses this file." % TAU_MIN]}
    _write_md(out_md, res)
    V._write_json(out, res)
    log("wrote %s and %s (%.0fs)" % (out, out_md, time.time() - t0))
    if not cal["check"]["ok"]:
        raise RelevanceError(_degenerate(cal, out))
    return res


def _degenerate(cal, out):
    return ("degenerate calibration in %s: tau = %s < %g, so %s of train_core crops (more than %g %%) are judged "
            "more non-plant than plant; the prompts do not separate genuine weeds from the non-plant set and tau "
            "would pass nearly every source. The file and its .md are written for a person to read; "
            "relevance.load refuses it" % (out, _fmt_p(cal.get("tau")), TAU_MIN,
                                           _fmt((cal.get("check") or {}).get("share_below_0.5"), 3),
                                           CAL_PERCENTILE))


# ------------------------------------------------------------------- use
def load(path, build_summary=None):
    """A relevance file, checked for use by an increment draw: its format, its
    rule's parameters, the calibration check (re-derived from tau; a failed
    one refuses), every source status re-derived from its recorded numbers,
    and (given select's build summary) that it was made for that build's
    increment pool and base_selected. A consistency check, not a seal: the
    file's sha256, which every draw records, identifies the file. Returns
    {path, sha256, tau, status: {source: status}, images: {source: images in
    the increment pool}, not_passing: {source: images}, rule}."""
    path = Path(os.path.abspath(str(path)))
    if not path.is_file():
        raise RelevanceError("no relevance file at %s (run inc.relevance build)" % path)
    try:
        with open(path) as fh:
            rel = json.load(fh)
    except (OSError, ValueError) as e:
        raise RelevanceError("%s cannot be read: %s" % (path, e))
    if not isinstance(rel, dict) or rel.get("format") != FORMAT:
        raise RelevanceError("%s is not a %s file" % (path, FORMAT))
    try:
        return _checked(path, rel, build_summary)
    except (KeyError, TypeError, AttributeError, ValueError) as e:
        raise RelevanceError("%s is malformed: %s: %s" % (path, type(e).__name__, e))


def _checked(path, rel, build_summary):
    """load()'s checks on a parsed file; a missing key or a wrong type raises
    (load turns it into a RelevanceError)."""
    prm = rel.get("params") or {}
    if (prm.get("min_crops") != MIN_CROPS or prm.get("calibration_percentile") != CAL_PERCENTILE
            or prm.get("tau_min") != TAU_MIN):
        raise RelevanceError("%s was made under another rule (min_crops %s, percentile %s, tau_min %s; this code: "
                             "%d, %g, %g)" % (path, prm.get("min_crops"), prm.get("calibration_percentile"),
                                              prm.get("tau_min"), MIN_CROPS, CAL_PERCENTILE, TAU_MIN))
    cal = rel.get("calibration") or {}
    tau = cal.get("tau")
    if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not 0.0 <= float(tau) <= 1.0:
        raise RelevanceError("%s records no calibrated tau" % path)
    chk = cal.get("check") or {}
    if not isinstance(chk.get("ok"), bool) or chk["ok"] != check_of(tau) or chk.get("tau_min") != TAU_MIN:
        raise RelevanceError("%s: the calibration check (%r) does not follow from tau %r under the rule (tau >= %g)"
                             % (path, chk.get("ok"), tau, TAU_MIN))
    if not check_of(tau):
        raise RelevanceError(_degenerate(cal, path))
    for t in TABLES:
        blk = rel.get(t) or {}
        src = blk.get("sources")
        if not isinstance(src, dict):
            raise RelevanceError("%s has no %s table" % (path, t))
        bad = sorted(s for s, e in src.items()
                     if status_of(e.get("crops_usable", 0), e.get("p_plant_median"), tau) != e.get("status")
                     or bool(e.get("passes")) != (e.get("status") == PASS))
        if bad:
            raise RelevanceError("%s: the %s status of %d source(s) does not follow from its recorded numbers "
                                 "under the rule, e.g. %s" % (path, t, len(bad), bad[:3]))
        want = {s: e["images"] for s, e in src.items() if e["status"] != PASS}
        if blk.get("not_passing") != want:
            raise RelevanceError("%s: the %s not_passing list is not its sources' statuses" % (path, t))
    if build_summary is not None:
        outs = build_summary.get("outputs") or {}
        for t in TABLES:
            got = ((rel.get("inputs") or {}).get(t) or {}).get("sha256")
            if got != (outs.get(TABLE_FILE[t]) or {}).get("sha256"):
                raise RelevanceError("%s was made for another %s (%s) than this select build's" %
                                     (path, TABLE_FILE[t], str(got)[:12]))
    pool = rel["increment_pool"]["sources"]
    return {"path": str(path), "sha256": C.sha256_file(path), "tau": float(tau),
            "status": {s: e["status"] for s, e in pool.items()},
            "images": {s: e["images"] for s, e in pool.items()},
            "not_passing": dict(rel["increment_pool"]["not_passing"]), "rule": rel.get("rule")}


# ---------------------------------------------------------------- outputs
def _fmt(x, digits=4):
    return "-" if x is None else "%.*f" % (digits, x)


def _fmt_p(x):
    """A P(plant) value for a person: tau, the medians, quartiles and
    percentiles all use it, so they stay comparable. At a logit scale near 100
    the values pile up next to 0 and 1, so the tails keep 3 significant digits
    (of p near 0, of 1 - p near 1)."""
    if x is None:
        return "-"
    x = float(x)
    if 1e-3 <= x <= 1.0 - 1e-3:
        return "%.4f" % x
    if x < 1e-3:
        return "%.3g" % x
    return "1 - %.3g" % (1.0 - x) if x < 1.0 else "1"


def _table_md(blk, tau):
    L = ["| Source | Images | Crops usable | Sampled | Median P(plant) | Share >= tau | "
         "Top prompt set: plant / non-plant / leaf disease | Status |",
         "|---|---|---|---|---|---|---|---|"]
    order = sorted(blk["sources"].items(), key=lambda kv: (-kv[1]["images"], kv[0]))
    for s, e in order:
        ts = e["top_set_share"]
        L.append("| %s | %d | %d | %d | %s | %s | %s / %s / %s | %s |"
                 % (s, e["images"], e["crops_usable"], e["sampled"], _fmt_p(e["p_plant_median"]),
                    _fmt(e["share_at_or_above_tau"], 3), _fmt(ts.get(PLANT), 2), _fmt(ts.get(NON_PLANT), 2),
                    _fmt(ts.get(DIAGNOSTIC), 2), e["status"]))
    return L


def _write_md(path, res):
    cal, te, prm = res["calibration"], res["text_encoder"], res["params"]
    ip, bs = res["increment_pool"], res["base_selected"]
    tau = cal["tau"]
    chk = cal["check"]
    L = ["# Source relevance (INC Step 1)", ""]
    if not chk["ok"]:
        L += ["**Calibration check FAILED: relevance.load refuses this file.** tau = %s < %g: %s of train_core "
              "crops score P(plant) below 0.5. The statuses below are for reading only." %
              (_fmt_p(tau), chk["tau_min"], _fmt(chk["share_below_0.5"], 3)), ""]
    L += ["Zero-shot %s, logit scale %.4f. P(plant) of a crop = the softmax over the %d plant and non-plant "
          "prompts, summed over the %d plant prompts. Calibrated on %d train_core crops: tau = the %gth "
          "percentile = %s (share of train_core crops at or above it: %s). Calibration check tau >= %g: %s "
          "(share of train_core crops below 0.5: %s)."
          % (te["name"], te["logit_scale"], len(PROMPTS[PLANT]) + len(PROMPTS[NON_PLANT]), len(PROMPTS[PLANT]),
             cal["crops"], cal["percentile"], _fmt_p(tau), _fmt(cal["share_at_or_above_tau"], 3), chk["tau_min"],
             "ok" if chk["ok"] else "FAILED", _fmt(chk["share_below_0.5"], 3)), ""]
    pv = te.get("provenance") or {}
    if pv:
        L += ["Model files: %s, snapshot %s; %s sha256 %s; tokenizer %s (context %s)." % (
            pv.get("repo"), pv.get("snapshot"), (pv.get("weights") or {}).get("file"),
            (pv.get("weights") or {}).get("sha256"), (pv.get("tokenizer") or {}).get("class"),
            (pv.get("tokenizer") or {}).get("context_length")), ""]
    L += ["Rule: %s. Sample: up to %d crops per source, seed %d." % (res["rule"], prm["sample"], prm["seed"]), "",
          "Prompts:", ""]
    for s in SETS:
        L.append("- %s%s: %s" % (s, " (reported only, outside the softmax)" if s == DIAGNOSTIC else "",
                                 "; ".join("'%s'" % p for p in PROMPTS[s])))
    np_ = ip["not_passing"]
    L += ["", "## Increment pool", "",
          "%d images in %d sources. Passing: %d sources, %d images. Not passing (excluded from the verified and "
          "OTHER_HEAVY increments): %d failing and %d insufficient sources, %d images%s."
          % (ip["images"], len(ip["sources"]), ip["sources_by_status"][PASS], ip["images_by_status"][PASS],
             ip["sources_by_status"][FAIL], ip["sources_by_status"][INSUFFICIENT], ip["not_passing_images"],
             (": " + ", ".join("%s (%d)" % kv for kv in sorted(np_.items(), key=lambda kv: (-kv[1], kv[0]))))
             if np_ else ""), ""]
    L += _table_md(ip, tau)
    nc = ip["name_check"]["sources"]
    L += ["", "### Name check (reported only)", "",
          "Non-passing sources whose name holds a plant word (%s): possible false negatives, since tau is "
          "calibrated on train_core (cwd12) and weeds from other cameras or domains can score lower. For a person "
          "to look at; it changes no status." % ", ".join(PLANT_NAME_WORDS), ""]
    L += (["| Source | Words | Images | Median P(plant) | Status |", "|---|---|---|---|---|"]
          + ["| %s | %s | %d | %s | %s |" % (s, ", ".join(e["words"]), e["images"], _fmt_p(e["p_plant_median"]),
                                            e["status"])
             for s, e in sorted(nc.items(), key=lambda kv: (-kv[1]["images"], kv[0]))]
          if nc else ["None."])
    L += ["", "## base_selected (for the record; base B is not rebuilt)", "",
          "%d images in %d sources; not passing: %s." % (
              bs["images"], len(bs["sources"]),
              ", ".join("%s (%d)" % kv for kv in sorted(bs["not_passing"].items(), key=lambda kv: (-kv[1], kv[0])))
              or "none"), ""]
    L += _table_md(bs, tau)
    L += ["", "## train_core P(plant)", "",
          "| Percentile | " + " | ".join(str(q) for q in DIST_PCTS) + " |",
          "|---|" + "---|" * len(DIST_PCTS),
          "| P(plant) | " + " | ".join(_fmt_p(cal["distribution"]["p%d" % q]) for q in DIST_PCTS) + " |", "",
          "Top prompt set on train_core: plant %s, non-plant %s, leaf disease %s."
          % tuple(_fmt(cal["top_set_share"][s], 3) for s in SETS), "",
          "## Notes", ""] + ["- " + n for n in res["notes"]] + [""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write("\n".join(L))
    os.replace(tmp, path)


# --------------------------------------------------------------------- CLI
def build_parser():
    ap = argparse.ArgumentParser(prog="python -m weed_optimizer_framework.tools.inc.relevance",
                                 description="INC Step 1: zero-shot BioCLIP-2 relevance of each harvested source")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="calibrate on train_core, judge every source, write relevance.json")
    b.add_argument("--sample", type=int, default=SAMPLE,
                   help="crops sampled per source (default %%(default)s; at least %d)" % MIN_CROPS)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--base-dir", default=None,
                   help="the inc.select build to judge (default INC_DIR/step1)")
    b.add_argument("--out", default=None, help="default <base-dir>/%s; the .md goes beside it" % OUT_NAME)
    b.add_argument("--force", action="store_true", help="overwrite a relevance file from another build")
    return ap


def main(argv=None, text_encoder=None):
    a = build_parser().parse_args(argv)
    try:
        build(a.sample, a.seed, a.base_dir, a.out, text_encoder=text_encoder, force=a.force)
    except (RelevanceError, S.SelectError) as e:
        print("inc.relevance: error: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
