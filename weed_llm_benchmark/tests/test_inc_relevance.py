#!/usr/bin/env python3
"""INC Step 1 source relevance (inc/relevance.py): zero-shot BioCLIP-2
P(plant) per crop, tau = the 5th percentile of train_core's P(plant), and a
source passes when the median P(plant) of its seeded sample is >= tau.

The world is synthetic, written in the formats relevance reads: verify's
crops.csv and emb_sNNN_of_NNN.npz shards, and an inc/select.py build
(select_summary.json naming increment_pool.jsonl, base_selected.jsonl,
crops.csv, the shard files and the embedding record). The text tower is
injected: a fake encoder maps each prompt of relevance.PROMPTS to a fixed
direction (plant prompts near e_plant, 'a photo of bare soil' on e_soil,
'a screenshot of a video game' on e_game, the leaf-disease prompts near
e_leaf), unnormalised, with logit scale 20.
  * train_core crops: e_plant plus up to CORE_SOIL (1.1) e_soil plus noise,
    so their P(plant) spreads: the 5th percentile is a real threshold, and
    it stays above 0.5 (the calibration check).
  * increment pool sources: weedA (450 plant crops, more than the sample),
    game (video-game crops), mixed (60 % plant, 40 % game), mostly_game (40 %
    plant), leafy (leaf-disease-looking plants), tiny (18 plant crops),
    nanny (25 crops, 10 of them without a feature), edge20 (exactly 20 plant
    crops), nocrop (3 images, every box too small to crop) and cropgame
    (video-game crops under a name with a plant word).
  * base_selected: weedA and game images of its own.
  * crops of images that were not admitted, and cwd12 copies, are in
    crops.csv but in neither manifest.

Pinned:
  * P(plant) recomputed here independently (softmax of 20 * cosine over the
    plant and non-plant prompts, summed over the plant prompts); tau is its
    5th percentile over every train_core crop with a feature; about 95 % of
    train_core crops pass it; the calibration check (tau >= 0.5) is recorded;
  * statuses: weedA, mixed, leafy and edge20 pass, game, mostly_game and
    cropgame fail, tiny, nanny and nocrop are insufficient; not_passing lists
    every non-passing source with its increment-pool images; each table uses
    only its own manifest's images;
  * the name check (reported only) lists exactly the non-passing sources
    whose name holds a plant word (cropgame, nocrop), not the passing weedA
    or leafy; relevance.md shows it, and prints tau and the medians in one
    format that keeps values near 0 apart;
  * a degenerate calibration fails closed: an encoder whose soil prompt also
    matches the train_core crops puts tau below 0.5; the build writes the
    file and its .md (flagged) and exits with an error, a rerun does too,
    and relevance.load refuses the file;
  * the sample: at most --sample crops with a feature, the rule's seeded draw
    (recomputed here), and a source median taken over it;
  * the leaf-disease set is reported (leafy's top prompts) and cannot change
    P(plant): an encoder that moves only those prompts gives the same numbers;
  * determinism, the no-op rerun, --force, and refusals before anything is
    written: an edited increment pool, crops.csv or shard, a text tower of
    another model, a wrong dimension, a non-finite feature, a sample below
    MIN_CROPS, an --out that is not .json;
  * relevance.load re-derives every status and the calibration check, and
    refuses a status or a check that does not follow from the recorded
    numbers, an edited not_passing list, another rule, a file made for
    another increment pool, and a malformed file (a RelevanceError, not a
    KeyError);
  * the real text tower's pinning (BioclipTextEncoder) against a fake
    open_clip module (a tiny torch model; skipped without torch): the
    tokenizer must be the one the model's config names (open_clip's
    HFTokenizer fallback and a wrong context length are refused), the config
    and weights must resolve from one hub snapshot, and the snapshot commit
    and the files' sha256 are recorded;
  * the CLI.

No open_clip (a fake module stands in for it), no network, no image file
is opened.

Run:  python3 tests/test_inc_relevance.py
"""
import csv
import json
import os
import pathlib
import shutil
import sys
import tempfile

TMP = pathlib.Path(tempfile.mkdtemp(prefix="inc_relevance_"))
os.environ["INC_DIR"] = str(TMP / "inc")          # never the machine's real INC_DIR
os.environ["REPO"] = str(TMP / "repo")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import relevance as REL  # noqa: E402
from weed_optimizer_framework.tools.inc import select as S  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, detail))
        FAILURES.append(name)


def raises(fn, contains=None):
    """The RelevanceError message, or None if fn did not raise one."""
    try:
        fn()
    except REL.RelevanceError as e:
        if contains and contains not in str(e):
            print("       raised without %r: %s" % (contains, e))
            return None
        return str(e)
    return None


# ------------------------------------------------------------------ the world
D = 16
SCALE = 20.0
RNG = np.random.default_rng(5)
BASIS = np.linalg.qr(RNG.normal(size=(D, D)))[0]
E_PLANT, E_GAME, E_SOIL, E_LEAF = BASIS[0], BASIS[1], BASIS[2], BASIS[3]
NOISE_DIRS = BASIS[8:]


def prompt_vectors(leaf_shift=0.0, soil_pull=0.0):
    """{prompt: raw text feature}. Plant prompts sit near e_plant, the leaf
    prompts near e_leaf (moved by leaf_shift), the other non-plant prompts on
    their own directions; soil_pull moves the soil prompt towards e_plant (a
    soil prompt that also matches seedling crops)."""
    out = {}
    for j, p in enumerate(REL.PROMPTS["plant"]):
        out[p] = 3.0 * (E_PLANT + 0.05 * BASIS[4 + j % 4])
    for j, p in enumerate(REL.PROMPTS["non_plant"]):
        v = {"a screenshot of a video game": E_GAME, "a photo of bare soil": E_SOIL + soil_pull * E_PLANT}.get(p)
        out[p] = 2.0 * (v if v is not None else BASIS[4 + j % 4] - 0.2 * E_PLANT)
    for j, p in enumerate(REL.PROMPTS["leaf_disease"]):
        out[p] = 1.5 * (E_LEAF + (0.1 + leaf_shift) * BASIS[5 + j % 3])
    return out


class FakeEncoder:
    def __init__(self, model="fake", vectors=None, dim=D, nan=False, scale=SCALE):
        self.model, self.name, self.logit_scale = model, "fake text tower", scale
        self.vectors = vectors or prompt_vectors()
        self.dim, self.nan, self.calls = dim, nan, []

    def __call__(self, prompts):
        self.calls.append(list(prompts))
        T = np.array([self.vectors[p][:self.dim] if self.dim <= D else
                      np.r_[self.vectors[p], np.zeros(self.dim - D)] for p in prompts])
        if self.nan:
            T[2, 0] = np.nan
        return T


def noise(n, s):
    return s * RNG.normal(size=(n, len(NOISE_DIRS))) @ NOISE_DIRS


def plant(n, soil_max=0.8):
    return E_PLANT + RNG.uniform(0, soil_max, size=(n, 1)) * E_SOIL + noise(n, 0.3)


def game(n):
    return E_GAME + 0.3 * E_PLANT * RNG.uniform(-1, 1, size=(n, 1)) + noise(n, 0.3)


# source -> (images, boxes per image, embedding maker(n) or None for no crop rows, NaN rows)
POOL_SOURCES = {
    "weedA": (150, 3, lambda n: plant(n), 0),
    "game": (60, 2, game, 0),
    "mixed": (50, 2, lambda n: np.r_[plant(60), game(40)][RNG.permutation(100)][:n], 0),
    "mostly_game": (50, 2, lambda n: np.r_[plant(40), game(60)][RNG.permutation(100)][:n], 0),
    "leafy": (40, 1, lambda n: 0.9 * E_PLANT + 1.2 * E_LEAF + noise(n, 0.2), 0),
    "tiny": (6, 3, lambda n: plant(n), 0),
    "nanny": (25, 1, lambda n: plant(n), 10),
    "edge20": (10, 2, lambda n: plant(n), 0),
    "nocrop": (3, 1, None, 0),
    "cropgame": (30, 1, game, 0),
}
BASE_SOURCES = {"weedA": (40, 2, lambda n: plant(n), 0), "game": (12, 2, game, 0)}
N_CORE = 400
CORE_SOIL = 1.1             # train_core crops carry up to this much e_soil: P(plant) spreads, p5 stays >= 0.5
EXPECT = {"weedA": REL.PASS, "game": REL.FAIL, "mixed": REL.PASS, "mostly_game": REL.FAIL,
          "leafy": REL.PASS, "tiny": REL.INSUFFICIENT, "nanny": REL.INSUFFICIENT,
          "edge20": REL.PASS, "nocrop": REL.INSUFFICIENT, "cropgame": REL.FAIL}
NAMED = {"cropgame": REL.FAIL, "nocrop": REL.INSUFFICIENT}     # non-passing, a plant word in the name


def make_world():
    """Step 1 and select outputs in their own formats under INC_DIR/step1.
    Returns ground truth."""
    step1 = pathlib.Path(V.STEP1)
    assert str(step1).startswith(str(TMP)), step1
    shutil.rmtree(step1, ignore_errors=True)
    (step1 / "labels").mkdir(parents=True)
    crops, emb = [], []                     # crops: (set, key, source, box, label)
    core_rows = []
    for i in range(N_CORE // 2):            # two boxes per train_core image
        key = "core_%04d" % i
        for b in range(2):
            crops.append(("core", key, "train_core", b, (i + b) % 12))
        emb.extend(E_PLANT + RNG.uniform(0, CORE_SOIL, size=(2, 1)) * E_SOIL + noise(2, 0.3))
        core_rows.append(key)
    crops.append(("core", "core_nan", "train_core", 0, 0))
    emb.append(np.full(D, np.nan))          # a train_core crop that failed to embed

    def rows_for(sources, prefix):
        rows = []
        for src, (n_img, nb, make, n_nan) in sources.items():
            X = make(n_img * nb) if make is not None else None
            t = 0
            for j in range(n_img):
                key = "%s_%s_%03d" % (prefix, src, j)
                lab = step1 / "labels" / (key + ".txt")
                C.write_yolo(lab, [(C.OTHER_PLANT, 0.5, 0.5, 0.2, 0.2)] * nb)
                rows.append({"image": "/x/%s.jpg" % key, "label": str(lab), "sha256": C.sha256_text(key),
                             "label_sha256": C.sha256_file(lab), "source": src, "session": "", "key": key})
                if X is None:
                    continue
                for b in range(nb):
                    crops.append(("pool", key, src, b, C.OTHER_PLANT))
                    emb.append(np.full(D, np.nan) if t < n_nan else X[t])
                    t += 1
        return rows

    pool = rows_for(POOL_SOURCES, "p")
    base = rows_for(BASE_SOURCES, "b")
    for r in range(10):                     # images verify did not admit: in neither manifest
        crops.append(("pool", "rejected_%02d" % r, "junk", 0, C.OTHER_PLANT))
        emb.append(game(1)[0])
    for r in range(3):                      # cwd12 copies: never judged here
        crops.append(("copy", "copy_%02d" % r, "copyslug", 0, r))
        emb.append(game(1)[0])

    with open(V.CROPS, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(V.CROP_FIELDS)
        for cid, (st, key, src, b, c) in enumerate(crops):
            wr.writerow([cid, st, key, "/x/%s.jpg" % key, src, src, b, "0.500000", "0.500000",
                         "0.200000", "0.200000", 640, 480, c, C.CLASS_NAMES[c]])
    crops_sha = C.sha256_file(V.CROPS)
    X = np.asarray(emb, dtype=np.float16)
    ids = np.arange(len(crops))
    pathlib.Path(V.EMB_DIR).mkdir(parents=True)
    emb_files = []
    for s, part in enumerate(np.array_split(RNG.permutation(len(ids)), 2)):
        p = pathlib.Path(V.EMB_DIR) / ("emb_s%03d_of_002.npz" % s)
        np.savez(p, meta=np.array(json.dumps({"crops_sha256": crops_sha, "embedder": "fake+fp16", "dim": D,
                                              "crops": int(len(part)), "stats": {}})),
                 crop_ids=ids[part], X=X[part])
        emb_files.append({"path": str(p), "sha256": C.sha256_file(p)})
    embeddings = {"nshards": 2, "embedder": "fake+fp16", "dim": D, "failed_crops": 0,
                  "nan_rows": int((~np.isfinite(X.astype(np.float32)).all(1)).sum())}
    outs = {}
    for name, rows in ((S.POOL, pool), (S.BASE_SELECTED, base)):
        outs[name] = {"sha256": C.write_manifest(step1 / name, rows), "rows": len(rows),
                      "path": str(step1 / name)}
    core_man = C.manifest_path("train_core")
    core_man.parent.mkdir(parents=True, exist_ok=True)
    core_man.write_text("{}\n")
    with open(step1 / S.SUMMARY, "w") as fh:
        json.dump({"outputs": outs, "crops": {"embeddings": embeddings},
                   "inputs": {"crops": {"path": str(V.CROPS), "sha256": crops_sha}, "emb_files": emb_files,
                              "train_core": {"path": str(core_man), "sha256": C.sha256_file(core_man)}}},
                  fh, indent=1, sort_keys=True)
    return {"crops": crops, "X": X, "pool": pool, "base": base}


def p_plant(X, idx):
    """P(plant), computed here from the definition, independently of relevance.py."""
    V_ = prompt_vectors()
    P = np.array([V_[p] for p in REL.PROMPTS["plant"]])
    N = np.array([V_[p] for p in REL.PROMPTS["non_plant"]])
    Tn = lambda A: A / np.linalg.norm(A, axis=1, keepdims=True)  # noqa: E731
    x = Tn(X[idx].astype(np.float64))
    lp, ln = SCALE * x @ Tn(P).T, SCALE * x @ Tn(N).T
    m = np.maximum(lp.max(1), ln.max(1))[:, None]
    ep, en = np.exp(lp - m).sum(1), np.exp(ln - m).sum(1)
    return ep / (ep + en)


COMMIT = "0123456789abcdef0123456789abcdef01234567"


def install_fake_open_clip(hub, tokenizer="simple", ctx=77, hf_name=None, weights=True, weights_commit=COMMIT):
    """A stand-in `open_clip` package in sys.modules: get_tokenizer,
    create_model_and_transforms (a tiny torch text model with open_clip's
    attributes), pretrained.download_pretrained_from_hf resolving files of a
    Hugging Face cache laid out under hub, and tokenizer.SimpleTokenizer /
    HFTokenizer. tokenizer: which class get_tokenizer returns ('simple', or
    'hf' as open_clip's fallback does). Returns the weights file."""
    import math
    import types
    import torch

    repo_dir = hub / "models--org--fake" / "snapshots"
    snap, wsnap = repo_dir / COMMIT, repo_dir / weights_commit
    snap.mkdir(parents=True, exist_ok=True)
    wsnap.mkdir(parents=True, exist_ok=True)
    text_cfg = {"context_length": 77}
    if hf_name:
        text_cfg["hf_tokenizer_name"] = hf_name
    (snap / "open_clip_config.json").write_text(json.dumps({"model_cfg": {"text_cfg": text_cfg}}))
    wfile = wsnap / "open_clip_model.safetensors"
    wfile.write_bytes(b"fake weights " + weights_commit.encode())

    class SimpleTokenizer:
        def __init__(self, context_length=77):
            self.context_length, self.vocab_size = context_length, 49408

        def __call__(self, texts):
            ids = [[49406] + [C.stable_int("%s/%d" % (t, i)) % 49000 for i in range(3)] + [49407] for t in texts]
            return torch.tensor([r + [0] * (self.context_length - len(r)) for r in ids], dtype=torch.long)

    class HFTokenizer:                                              # not a SimpleTokenizer subclass
        def __init__(self, context_length=77):
            self.context_length = context_length

        def __call__(self, texts):
            return SimpleTokenizer(self.context_length)(texts)

    class TextModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.context_length, self.vocab_size = 77, 49408
            self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(100.0)))
            torch.manual_seed(0)
            self.emb = torch.nn.Embedding(49408, 8)

        def encode_text(self, tokens, normalize=False):
            return self.emb(tokens).sum(1)

    def download_pretrained_from_hf(model_id, filename=None, revision=None, cache_dir=None):
        if filename is None:
            if not weights:
                raise FileNotFoundError("Failed to download file (open_clip_pytorch_model.bin) for %s" % model_id)
            return str(wfile)
        return str(snap / filename)

    tok = HFTokenizer(ctx) if tokenizer == "hf" else SimpleTokenizer(ctx)
    oc = types.ModuleType("open_clip")
    oc.__version__ = "fake"
    oc.get_tokenizer = lambda name: tok
    oc.create_model_and_transforms = lambda name: (TextModel(), None, None)
    pre = types.ModuleType("open_clip.pretrained")
    pre.download_pretrained_from_hf = download_pretrained_from_hf
    tk = types.ModuleType("open_clip.tokenizer")
    tk.SimpleTokenizer, tk.HFTokenizer = SimpleTokenizer, HFTokenizer
    oc.pretrained, oc.tokenizer = pre, tk
    sys.modules.update({"open_clip": oc, "open_clip.pretrained": pre, "open_clip.tokenizer": tk})
    return wfile


def test_text_tower():
    """BioclipTextEncoder's pinning, against a fake open_clip."""
    print("the text tower is pinned to the model's own files (fake open_clip)")
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  skip (no torch)")
        return
    from weed_optimizer_framework.tools import semisup_labeler as SL
    saved_device = SL._device
    SL._device = lambda: "cpu"
    name = "hf-hub:org/fake"
    prompts = REL.prompt_table()[0]
    try:
        hub = TMP / "hub_ok"
        wfile = install_fake_open_clip(hub)
        enc = REL.BioclipTextEncoder(name)
        T = REL.text_features(enc, prompts, 8)
        pv = enc.provenance
        check("the model's own tokenizer (SimpleTokenizer, as the config names no hf_tokenizer_name) is accepted; "
              "the features are one row per prompt at the model's logit scale",
              T.shape == (len(prompts), 8) and abs(enc.logit_scale - 100.0) < 1e-3
              and pv["tokenizer"] == {"class": "SimpleTokenizer", "hf_tokenizer_name": None, "context_length": 77,
                                      "vocab_size": 49408}, pv)
        check("the snapshot commit and the sha256 of the config and the weights file open_clip loads are recorded",
              pv["snapshot"] == COMMIT and pv["repo"] == "org/fake"
              and pv["weights"] == {"file": wfile.name, "sha256": C.sha256_file(wfile),
                                    "bytes": wfile.stat().st_size}
              and pv["config"]["sha256"] == C.sha256_file(wfile.parent / "open_clip_config.json"), pv)
        install_fake_open_clip(TMP / "hub_hf", tokenizer="hf", hf_name="org/fake-tok")
        check("a config that names an hf_tokenizer_name expects HFTokenizer",
              REL.BioclipTextEncoder(name).provenance["tokenizer"]["class"] == "HFTokenizer")
        for j, (what, kw, contains) in enumerate((
                ("open_clip's fallback HFTokenizer when the config names none", {"tokenizer": "hf"},
                 "the config names SimpleTokenizer"),
                ("a tokenizer of another context length", {"ctx": 64}, "context length"),
                ("weights that do not resolve (open_clip would build a random model)", {"weights": False},
                 "do not resolve"),
                ("a config and weights of two snapshots", {"weights_commit": "f" * 40}, "one hub snapshot"))):
            install_fake_open_clip(TMP / ("hub_%d" % j), **kw)
            check("refuses %s" % what, raises(lambda: REL.BioclipTextEncoder(name), contains) is not None)
        check("refuses a model that is not an hf-hub one",
              raises(lambda: REL.BioclipTextEncoder("ViT-L-14"), "only hf-hub") is not None)
    finally:
        SL._device = saved_device
        for m in ("open_clip", "open_clip.pretrained", "open_clip.tokenizer"):
            sys.modules.pop(m, None)


def strip(res):
    return {k: v for k, v in res.items() if k not in ("built_utc", "seconds", "outputs")}


def main():
    W = make_world()
    step1 = pathlib.Path(V.STEP1)
    out = step1 / REL.OUT_NAME
    enc = FakeEncoder()
    res = REL.build(sample=300, seed=0, text_encoder=enc)
    crops = W["crops"]
    st = np.array([c[0] for c in crops])
    fin = np.isfinite(W["X"].astype(np.float32)).all(1)

    print("calibration")
    core = np.flatnonzero((st == "core") & fin)
    pc = p_plant(W["X"], core)
    tau = float(np.percentile(pc, 5))
    cal = res["calibration"]
    check("tau = the 5th percentile of P(plant) over every train_core crop with a feature (recomputed here)",
          abs(cal["tau"] - tau) < 1e-9 and cal["crops"] == N_CORE and cal["crops_without_feature"] == 1,
          (cal["tau"], tau, cal["crops"]))
    check("the threshold is a real one: train_core P(plant) spreads (p1 < tau < median), and tau >= 0.5",
          cal["distribution"]["p1"] < tau < cal["distribution"]["p50"] and REL.TAU_MIN <= tau < 0.99,
          cal["distribution"])
    check("the calibration check is recorded: ok, tau_min 0.5, the share of train_core crops below 0.5",
          cal["check"]["ok"] is True and cal["check"]["tau_min"] == 0.5 == res["params"]["tau_min"]
          and abs(cal["check"]["share_below_0.5"] - float(np.mean(pc < 0.5))) < 1e-6
          and cal["check"]["share_below_0.5"] <= 0.05, cal["check"])
    check("about 95 % of train_core crops score at or above tau",
          abs(cal["share_at_or_above_tau"] - 0.95) <= 1.0 / N_CORE + 1e-9, cal["share_at_or_above_tau"])
    check("the distribution is recorded (percentiles, histogram over every crop)",
          sum(cal["histogram"]["counts"]) == N_CORE and set(cal["distribution"]) == {
              "p%d" % q for q in REL.DIST_PCTS})

    print("sources")
    ip, bs = res["increment_pool"], res["base_selected"]
    got = {s: e["status"] for s, e in ip["sources"].items()}
    check("statuses: plant sources pass, non-plant sources fail, fewer than %d usable crops is insufficient"
          % REL.MIN_CROPS, got == EXPECT, got)
    check("passes is status == pass", all(e["passes"] == (e["status"] == REL.PASS) for e in ip["sources"].values()))
    check("images per source are the increment pool's",
          {s: e["images"] for s, e in ip["sources"].items()} == {s: v[0] for s, v in POOL_SOURCES.items()}
          and ip["images"] == len(W["pool"]))
    want_np = {s: POOL_SOURCES[s][0] for s, v in EXPECT.items() if v != REL.PASS}
    check("not_passing = every failing or insufficient source with its increment-pool images",
          ip["not_passing"] == want_np and ip["not_passing_images"] == sum(want_np.values())
          and ip["sources_by_status"] == {"pass": 4, "fail": 3, "insufficient": 3}, ip["not_passing"])
    e = ip["sources"]
    check("usable crops: NaN rows are not usable; a source without crop rows has none",
          e["nanny"]["crops"] == 25 and e["nanny"]["crops_usable"] == 15 and e["tiny"]["crops_usable"] == 18
          and e["edge20"]["crops_usable"] == 20 and e["nocrop"]["crops"] == 0
          and e["nocrop"]["p_plant_median"] is None, (e["nanny"]["crops_usable"], e["nocrop"]))
    check("insufficient sources are still measured when they have crops (tiny's median is plant-like)",
          e["tiny"]["p_plant_median"] > tau and e["nanny"]["p_plant_median"] > tau)

    # the sample, recomputed from the rule
    keys_of = {r["key"]: r["source"] for r in W["pool"]}
    ids = [i for i, c in enumerate(crops) if c[0] == "pool" and keys_of.get(c[1]) == "weedA" and fin[i]]
    rng = np.random.default_rng(C.stable_int(REL.SEED_TEXT % (0, "increment_pool", "weedA")))
    pick = np.array(ids)[np.sort(rng.choice(len(ids), size=300, replace=False))]
    med = float(np.median(p_plant(W["X"], pick)))
    check("sample: at most --sample usable crops, the rule's seeded draw; median over it (recomputed here)",
          e["weedA"]["crops_usable"] == 450 and e["weedA"]["sampled"] == 300
          and abs(e["weedA"]["p_plant_median"] - med) < 1e-9, (e["weedA"]["p_plant_median"], med))
    ids_g = [i for i, c in enumerate(crops) if c[0] == "pool" and keys_of.get(c[1]) == "game"]
    check("a source under --sample uses every usable crop",
          e["game"]["sampled"] == 120 == len(ids_g)
          and abs(e["game"]["p_plant_median"] - float(np.median(p_plant(W["X"], ids_g)))) < 1e-9)
    check("mixed: median plant, share at or above tau ~0.6; mostly_game: ~0.4, fails",
          0.5 < e["mixed"]["share_at_or_above_tau"] < 0.7 and 0.3 < e["mostly_game"]["share_at_or_above_tau"] < 0.5,
          (e["mixed"]["share_at_or_above_tau"], e["mostly_game"]["share_at_or_above_tau"]))
    check("the video-game source's top prompt is the video-game prompt",
          e["game"]["top_set_share"]["non_plant"] > 0.9
          and max(e["game"]["top_prompts"], key=e["game"]["top_prompts"].get) == "a screenshot of a video game",
          e["game"]["top_prompts"])
    check("leaf-disease diagnostic: leafy's top prompts are leaf-disease prompts, and it passes",
          e["leafy"]["top_set_share"]["leaf_disease"] > 0.9 and e["leafy"]["status"] == REL.PASS,
          e["leafy"]["top_set_share"])
    check("base_selected is judged on its own images (game fails there too) and never enters not_passing of the "
          "increment pool",
          {s: x["status"] for s, x in bs["sources"].items()} == {"weedA": REL.PASS, "game": REL.FAIL}
          and bs["sources"]["weedA"]["images"] == 40 and bs["not_passing"] == {"game": 12}
          and "junk" not in ip["sources"] and "copyslug" not in ip["sources"])
    check("prompts, the rule and the text encoder are recorded",
          res["params"]["prompts"] == {s: list(v) for s, v in REL.PROMPTS.items()}
          and res["text_encoder"]["logit_scale"] == SCALE and res["text_encoder"]["model"] == "fake"
          and res["params"]["text_model"] == "fake" and res["rule"] == REL.RULE
          and enc.calls == [REL.prompt_table()[0]])
    md = (step1 / "relevance.md").read_text()
    check("relevance.md is written beside relevance.json", "| game | 60 |" in md)
    nc = ip["name_check"]
    check("name check (reported only): exactly the non-passing sources with a plant word in the name, with "
          "their status (not the passing weedA or leafy); in the .md",
          {s: x["status"] for s, x in nc["sources"].items()} == NAMED
          and nc["sources"]["cropgame"]["words"] == ["crop"] and nc["words"] == list(REL.PLANT_NAME_WORDS)
          and "### Name check (reported only)" in md and "| cropgame | crop | 30 |" in md, nc)
    check("relevance.md prints tau and every median in one format that keeps values near 0 apart "
          "(game's median is not '0.0000')",
          REL._fmt_p(e["game"]["p_plant_median"]) in md and REL._fmt_p(e["game"]["p_plant_median"]) != "0.0000"
          and e["game"]["p_plant_median"] < 1e-3 and "percentile = %s" % REL._fmt_p(tau) in md
          and REL._fmt_p(1 - 2.5e-7) == "1 - 2.5e-07" and REL._fmt_p(2.5e-7) == "2.5e-07"
          and REL._fmt_p(0.61234) == "0.6123", (REL._fmt_p(e["game"]["p_plant_median"]),))

    print("the diagnostic set cannot move P(plant)")
    enc2 = FakeEncoder(vectors=prompt_vectors(leaf_shift=3.0))
    r2 = REL.build(sample=300, seed=0, text_encoder=enc2, out=TMP / "diag" / "relevance.json")
    same = all(r2["increment_pool"]["sources"][s]["p_plant_median"] == x["p_plant_median"]
               and r2["increment_pool"]["sources"][s]["status"] == x["status"] for s, x in e.items())
    check("an encoder that moves only the leaf-disease prompts gives the same P(plant), tau and statuses",
          same and r2["calibration"]["tau"] == cal["tau"]
          and r2["increment_pool"]["sources"]["leafy"]["top_set_share"] != e["leafy"]["top_set_share"])

    print("a degenerate calibration fails closed")
    deg = TMP / "degenerate" / "relevance.json"
    err = raises(lambda: REL.build(sample=300, seed=0, text_encoder=FakeEncoder(vectors=prompt_vectors(soil_pull=0.5)),
                                   out=deg), "degenerate calibration")
    dj = json.loads(deg.read_text()) if deg.exists() else {}
    dmd = deg.with_suffix(".md").read_text() if deg.with_suffix(".md").exists() else ""
    dcal = dj.get("calibration") or {"check": {}}
    check("a soil prompt that also matches the train_core crops puts tau below 0.5: the build writes the file and "
          "its .md (flagged) for a person, then raises",
          err is not None and dcal["tau"] < REL.TAU_MIN and dcal["check"]["ok"] is False
          and dcal["check"]["share_below_0.5"] > 0.05 and "Calibration check FAILED" in dmd, (err, dcal.get("tau")))
    check("a rerun with the same inputs and parameters raises again (no silent no-op)",
          raises(lambda: REL.build(sample=300, seed=0, out=deg,
                                   text_encoder=FakeEncoder(vectors=prompt_vectors(soil_pull=0.5))),
                 "degenerate calibration") is not None)
    check("relevance.load refuses it",
          "degenerate calibration" in (raises(lambda: REL.load(deg, json.loads((step1 / S.SUMMARY).read_text())))
                                       or ""))
    rc_deg = REL.main(["build", "--out", str(TMP / "degenerate_cli" / "relevance.json")],
                      text_encoder=FakeEncoder(vectors=prompt_vectors(soil_pull=0.5)))
    check("the CLI exits 2", rc_deg == 2 and (TMP / "degenerate_cli" / "relevance.json").exists())

    print("determinism and reruns")
    r3 = REL.build(sample=300, seed=0, text_encoder=FakeEncoder(), out=TMP / "again" / "relevance.json")
    check("same inputs and parameters: the same record", strip(r3) == strip(res))
    r4 = REL.build(sample=300, seed=1, text_encoder=FakeEncoder(), out=TMP / "seed1" / "relevance.json")
    check("another seed: another weedA sample, the same statuses",
          r4["increment_pool"]["sources"]["weedA"]["p_plant_median"] != e["weedA"]["p_plant_median"]
          and {s: x["status"] for s, x in r4["increment_pool"]["sources"].items()} == got)
    built = json.loads(out.read_text())["built_utc"]
    enc5 = FakeEncoder()
    REL.build(sample=300, seed=0, text_encoder=enc5)
    check("a rerun with the same inputs and parameters is a no-op (no model call)",
          json.loads(out.read_text())["built_utc"] == built and enc5.calls == [])
    check("a rerun with other parameters over the file refuses without --force",
          raises(lambda: REL.build(sample=200, seed=0, text_encoder=FakeEncoder()), "--force") is not None)
    REL.build(sample=200, seed=0, text_encoder=FakeEncoder(), force=True)
    check("--force overwrites", json.loads(out.read_text())["params"]["sample"] == 200)
    REL.build(sample=300, seed=0, text_encoder=FakeEncoder(), force=True)

    print("refusals")
    bad = TMP / "bad" / "relevance.json"

    def refused(what, fn, contains):
        err = raises(fn, contains)
        check("refuses %s" % what, err is not None and not bad.exists(), err)

    for what, kw, contains in (
            ("a text tower of another model", {"text_encoder": FakeEncoder(model="other")}, "image tower"),
            ("text features of another dimension", {"text_encoder": FakeEncoder(dim=D - 1)}, "dim"),
            ("a non-finite text feature", {"text_encoder": FakeEncoder(nan=True)}, "non-finite"),
            ("a non-positive logit scale", {"text_encoder": FakeEncoder(scale=0.0)}, "logit scale"),
            ("a sample below MIN_CROPS", {"sample": REL.MIN_CROPS - 1}, "--sample"),
            ("a negative seed", {"seed": -1}, "--seed")):
        args = dict(sample=300, seed=0, text_encoder=FakeEncoder(), out=bad)
        args.update(kw)
        refused(what, lambda a=args: REL.build(**a), contains)
    refused("an --out that is not .json",
            lambda: REL.build(text_encoder=FakeEncoder(), out=TMP / "bad" / "relevance.txt"), ".json")
    for name, contains in ((S.POOL, "changed since the select build"), (V.CROPS, "not the one"),
                           (pathlib.Path(V.EMB_DIR) / "emb_s001_of_002.npz", "not the one the select build")):
        p = step1 / name if not os.path.isabs(str(name)) else pathlib.Path(name)
        saved = p.read_bytes()
        p.write_bytes(saved + b"\n")
        try:
            refused("an edited %s" % p.name, lambda: REL.build(text_encoder=FakeEncoder(), out=bad), contains)
        finally:
            p.write_bytes(saved)

    print("load")
    sel = json.loads((step1 / S.SUMMARY).read_text())
    rel = REL.load(out, sel)
    check("load: the increment pool's statuses and images, tau and the file's sha256",
          rel["status"] == got and rel["images"] == {s: v[0] for s, v in POOL_SOURCES.items()}
          and rel["tau"] == cal["tau"] and rel["sha256"] == C.sha256_file(out) and rel["not_passing"] == want_np)
    edited = TMP / "edited" / "relevance.json"
    edited.parent.mkdir(parents=True)

    def edit(fn, src=out):
        d = json.loads(src.read_text())
        fn(d)
        edited.write_text(json.dumps(d))
        try:
            REL.load(edited, sel)
        except REL.RelevanceError as e:
            return str(e)
        except Exception as e:  # noqa: BLE001 - reported as the check's failure
            return "raised %s, not a RelevanceError" % type(e).__name__
        return None

    def flip(d):
        d["increment_pool"]["sources"]["game"].update(status="pass", passes=True)
        del d["increment_pool"]["not_passing"]["game"]

    check("load refuses a status that does not follow from its recorded numbers (game marked pass)",
          "does not follow" in (edit(flip) or ""))
    check("load refuses a not_passing list that is not the statuses",
          "not_passing" in (edit(lambda d: d["increment_pool"]["not_passing"].pop("tiny")) or ""))
    check("load refuses a tau raised above a passing source's median (its status no longer follows)",
          "does not follow" in (edit(lambda d: d["calibration"].update(tau=0.999999)) or ""))
    check("load refuses a file made under another rule (min_crops, tau_min)",
          "another rule" in (edit(lambda d: d["params"].update(min_crops=5)) or "")
          and "another rule" in (edit(lambda d: d["params"].update(tau_min=0.1)) or ""))
    check("load refuses a tau lowered below 0.5 while the check still says ok (the check does not follow)",
          "calibration check" in (edit(lambda d: d["calibration"].update(tau=0.01)) or ""))
    check("load refuses the degenerate file with its check flipped to ok (the check does not follow)",
          "calibration check" in (edit(lambda d: d["calibration"]["check"].update(ok=True), src=deg) or ""))
    check("load refuses a file without a calibration check",
          "calibration check" in (edit(lambda d: d["calibration"].pop("check")) or ""))

    def no_images(d):
        del d["increment_pool"]["sources"]["weedA"]["images"]
    for what, fn in (("a source entry without images", no_images),
                     ("a table without not_passing", lambda d: d["increment_pool"].pop("not_passing")),
                     ("a source entry that is not an object",
                      lambda d: d["increment_pool"]["sources"].update(weedA=[1, 2])),
                     ("a non-numeric usable-crop count",
                      lambda d: d["increment_pool"]["sources"]["weedA"].update(crops_usable="many"))):
        msg = edit(fn)
        check("load refuses a malformed file (%s) with a RelevanceError" % what,
              msg is not None and not msg.startswith("raised "), msg)
    other = dict(sel, outputs=dict(sel["outputs"], **{S.POOL: dict(sel["outputs"][S.POOL], sha256="0" * 64)}))
    check("load refuses a file made for another increment pool",
          "another" in (raises(lambda: REL.load(out, other)) or ""))
    check("load refuses a missing file", raises(lambda: REL.load(TMP / "none.json")) is not None)

    print("CLI")
    rc = REL.main(["build", "--out", str(TMP / "cli" / "relevance.json")], text_encoder=FakeEncoder())
    rc_bad = REL.main(["build", "--sample", "5", "--out", str(TMP / "cli2" / "relevance.json")],
                      text_encoder=FakeEncoder())
    check("CLI build (exit 0) = the same record; a bad --sample exits 2",
          rc == 0 and strip(json.loads((TMP / "cli" / "relevance.json").read_text())) == strip(res)
          and rc_bad == 2 and not (TMP / "cli2" / "relevance.json").exists())


if __name__ == "__main__":
    try:
        main()
        test_text_tower()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    print("\n%d failure(s)" % len(FAILURES))
    sys.exit(1 if FAILURES else 0)
