#!/usr/bin/env python3
"""The funnel audit end to end, F3 to F9, on one synthetic world (docs/
FUNNEL_AUDIT.md §10; docs/FUNNEL_AUDIT_RUNNER.md §5.6.5, §6, §7.3), through
the command line (funnel/__main__.py main) with fake models injected, then
the realloop_v2 build that consumes the recovery (inc/realloop.py, recovered
mode) and the arms recover builds from it.

Chain: fetch taxonomy -> census -> (realloop_v1 build) -> fetch kt7 ->
arrival check -> leak -> (cards) -> map geometry -> embed-judges -> qualify
-> map relation -> draw -> sheets -> rl-b -> (RL-A answers) -> ingest ->
qualify --rl -> (DA record) -> estimate -> recover -> (realloop_v2 build)
-> recover --arms. Steps in parentheses are not funnel verbs; the rest run
through the CLI.

The world is tests/funnel_world.py's Step 1 world (verify pool, crops, embed,
fit, calibrate, admit and select run on painted pictures: every box is painted
in the colour of its TRUE class), grown by a plant hook:
  * 9 train_core pictures per capture session, and copies of them in the two
    reference-copy sources and cottonweed_sp8 so that each holds one class of
    >= 20 boxes (the relation audit's H0(b) units);
  * 80 vetoed images in the authoritative source weed_crop: a verified
    Waterhemp box, a Blackbean-labelled box painted Palmer amaranth (the
    conflict that vetoes the image) and a Kochia box (G2v, policy R-V);
  * 40 numeric-source images whose id-3 box is painted Purslane (a
    no-information conflict stratum, policy R-J);
  * 30 vetoed images like the above in the licensed greenhouse source (their
    blocking box a Blackbean painted Morning glory), and 30 Purslane images
    like the above in a source the config marks not recoverable: the gates of
    their strata pass, so only the guards may keep them out;
  * six numeric-source pictures 7-10 dHash bits from a train_core picture but
    another photograph (the copy detector's calibration negatives), and a
    flipped copy of a dev picture in the greenhouse source (its plain dHash is
    far from the dev picture, so Step 1 kept it; the detector's variants find
    it, and the source is quarantined as a whole).
Its realloop_v1 is built by the real realloop builder (testing).

Fakes (no network, no GPU): the DINOv2 stand-in describes a crop tile by the
colour at its centre and a whole image by a chromaticity histogram (it
survives the H6 augmentation families); the zero-shot text tower points a
target's prompt at its class colour; the step-1 embedder reads the corner of
a KT7 photo (so the probe errs on the independent photos and the judges'
rescue can be measured); the web is the recorded GBIF answers plus a fake
iNaturalist serving painted photos; the cards are the fetched layout
(cards/index.json and upstream annotations built from the pool's own label
files); RL-B is a fake ollama server and RL-A a stub, both answering from the
painted colour of the crop tile they are shown and nothing else; the DA
record (prospective_da.json) is a synthetic fixture. The test copy of
prereg_v1.json has its sampling minimums set to 0, because the world's frames
are far smaller than the real pool's (draw refuses a frame below its
minimum).

Asserted, beyond every step completing:
  * the freshness chain (runner §7.3): every artifact records this prereg's
    core, the contract and the domain config; every input it records still
    hashes as recorded; its code record is the code that ran; each consumer
    records its producer's file; the sample lock names the sample, key,
    frames and name status;
  * census reconciles; leak (copy detector version 2: the prereg's
    amendment A2 requires it, so the verb writes leak_v2.json and no
    leak_v1.json) calibrates per image and quarantines the source holding
    the flipped dev copy whole; H0(b) passes on the planted copies; pairs stay on cluster sheets;
    no pool sheet names a source, unit, stratum or verdict; RL-A answers pool
    sheets only; the machine RL qualifies at species level in H0's scope;
  * the estimates cover the planted truth: H0 is supported with the planted
    share inside its interval, and every labelled box stratum's interval
    (events target, label, pred) covers the share of its frame units that
    the painted truth makes true;
  * the audit holds the per-stratum events the recovery gates read; the
    autopilot's H11 scorer agrees with the estimator's; D18 reads the audit
    and proposes only policies that recovered rows; remote funnel summary
    ships the aggregates with the audit unredacted and no row-level file;
  * recovery respects every guard: only strata whose gate passed recover,
    each passed gate Rogan-Gladen corrected above the prereg's bounds; no
    quarantined source (not even its clean images), no not-recoverable
    source, no domain-dev image, no base or non-pool image; every row clear
    of the never-train index on the unmasked and the masked image
    (recomputed); every recovered label the painted truth; masks one flat
    fill each; the step-1 labels unchanged;
  * realloop_v2 records the overlay and draws 6 unclean steps from its rows
    only; build-baseline's recovery provenance accepts every arm;
  * census, draw, estimate and recover rerun as no-ops;
  * unit regressions of the integration defects this test found (sheets'
    pred names, the Korn-Graubard n at 0 or 1, the "label:<class>" event,
    the DA purity row).

Run:  python3 tests/test_funnel_pipeline.py
"""
import collections
import csv
import io
import json
import os
import pathlib
import shutil
import socket
import sys
import tempfile
import time
import zlib

TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent
FAILURES = []
SKIPS = []
T0 = time.time()


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s %s" % (name, str(detail)[:2000]))
        FAILURES.append(name)


def skip(name, reason):
    print("  SKIP: %s (%s)" % (name, reason))
    SKIPS.append(name)


def finish():
    print("\n%d failure(s), %d skipped: %s" % (len(FAILURES), len(SKIPS), ", ".join(FAILURES + SKIPS) or "none"))
    sys.exit(1 if FAILURES else 0)


# ====================================================================== fakes
NB = 16


def chroma_hist(pil):
    """A saturation-weighted chromaticity histogram (r and g of r+g+b),
    soft-binned: it survives flips, rotations, crops, brightness, blur,
    shear, letterboxing and re-encoding, and tells two painted pictures
    apart."""
    import numpy as np
    a = np.asarray(pil.convert("RGB"), dtype=np.float64).reshape(-1, 3)
    s = a.sum(1)
    mx, mn = a.max(1), a.min(1)
    w = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0) * (s > 45)
    x = a[:, 0] / np.maximum(s, 1) * (NB - 1)
    y = a[:, 1] / np.maximum(s, 1) * (NB - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    x1, y1 = np.minimum(x0 + 1, NB - 1), np.minimum(y0 + 1, NB - 1)
    hist = np.zeros((NB, NB))
    for xi, yi, ww in ((x0, y0, (1 - fx) * (1 - fy)), (x1, y0, fx * (1 - fy)), (x0, y1, (1 - fx) * fy),
                       (x1, y1, fx * fy)):
        np.add.at(hist, (xi, yi), w * ww)
    h = hist.ravel()
    n = np.linalg.norm(h)
    return h / n if n > 0 else h


def centre_class(tile, colors):
    """The class a tile was painted as: the nearest class colour at its centre."""
    import numpy as np
    a = np.asarray(tile.convert("RGB"), dtype=np.float64)
    h, w = a.shape[:2]
    ctr = a[h // 2 - 12:h // 2 + 12, w // 2 - 12:w // 2 + 12].reshape(-1, 3).mean(0)
    return int(np.argmin(((np.array(colors, dtype=np.float64) - ctr) ** 2).sum(1)))


class FakeDino(object):
    """The DINOv2 stand-in (embed.Dinov2Embedder and embed.LazyEmbedder are
    replaced by it): a 224 x 224 crop tile is described by the colour at its
    centre (one-hot, with noise of its own); a whole-image view (shorter edge
    256) by its chromaticity histogram."""
    COLORS = None

    def __init__(self, model="facebook/dinov2-base", pooling="cls", *a, **k):
        self.model_name, self.pooling = model, pooling
        self.name = "%s:%s" % (model, pooling)
        self.dim = 24 + NB * NB

    def __call__(self, pils):
        import numpy as np
        out = []
        for p in pils:
            v = np.zeros(self.dim)
            if p.size == (224, 224):
                a = np.asarray(p.convert("RGB"))
                k = centre_class(p, self.COLORS)
                rng = np.random.default_rng(zlib.crc32(np.ascontiguousarray(a[96:128, 96:128]).tobytes()) + 7)
                v[k] = 1.0
                v[:24] += rng.normal(0, 1, 24) * 0.02
            else:
                v[24:] = chroma_hist(p)
            out.append(v)
        return np.stack(out).astype(np.float32)


class Kt7ProbeEmbedder(object):
    """The Step 1 embedder stand-in for the KT7 photos (adapter.bioclip_embedder):
    FW.FakeEmbedder's features, read from the top-left corner of the crop
    instead of its centre, so the step-1 probe misreads the independent
    photos (the probe is wrong under shift, contract §4.2) and the judges'
    rescue of its errors can be measured."""
    name = "fake-colour"
    dim = 24

    def __init__(self, fw):
        self.fw = fw

    def __call__(self, pils):
        import numpy as np
        from PIL import Image
        out = []
        for p in pils:
            a = np.asarray(p.convert("RGB"))
            corner = Image.fromarray(np.ascontiguousarray(a[8:32, 8:32])).resize((224, 224))
            out.append(self.fw.fake_feature(corner))
        return np.stack(out)


class FakeText(object):
    """The zero-shot text tower stand-in over the world's 24-dim crop
    features: a target's prompt points at its class colour, an attractor's or
    a named non-target's at the other class's, a non-object prompt at an
    unused axis."""
    name = "fake-text"
    logit_scale = 30.0
    provenance = "test"

    def __init__(self, domain):
        self.common = {t["common"].lower(): int(t["id"]) for t in domain.targets}
        self.other = int(domain.other["id"])

    def __call__(self, texts):
        import numpy as np
        out = []
        for t in texts:
            v = np.zeros(24)
            low = t.lower()
            hit = [cid for c, cid in self.common.items() if ("common name %s." % c) in low]
            if hit:
                v[hit[0]] = 1.0
            elif "common name" in low:
                v[self.other] = 1.0
            else:
                v[20] = 1.0
            out.append(v)
        return np.stack(out)


class Oracle(object):
    """The reference labeller stub: it answers a panel from the pixels it is
    shown, the painted colour of the crop tile being the truth. It never sees
    a unit id, a source, a stratum or a verdict."""

    def __init__(self, domain, colors):
        self.opts = domain.options()
        self.n_targets = len(domain.targets)
        tail = [o for o in self.opts if o["kind"] == "tail"]
        self.other_opt = tail[0]["n"]
        self.colors = colors
        self.calls = 0

    def answer_tile(self, tile):
        cls = centre_class(tile, self.colors)
        if 0 <= cls < self.n_targets:
            return "%d, YES" % (cls + 1)
        return "%d, YES" % self.other_opt

    @staticmethod
    def pair_answer(a, b):
        """1 (the same photograph) when the two tiles' histograms agree, else
        3 (different photographs)."""
        return 1 if float(chroma_hist(a) @ chroma_hist(b)) >= 0.95 else 3


def ollama_transport(oracle, model, digest="sha256:" + "0" * 64):
    """A fake ollama server behind rl.OllamaClient: vision capability, the
    tagged model, and /api/chat answered by the oracle from the last image
    (the panel cut from the sheet)."""
    import base64
    from PIL import Image

    def transport(method, url, body, timeout):
        path = url.split("//", 1)[-1].split("/", 1)[-1]
        if path == "api/show":
            return 200, json.dumps({"capabilities": ["completion", "vision"]}).encode()
        if path == "api/tags":
            return 200, json.dumps({"models": [{"name": model, "model": model, "digest": digest}]}).encode()
        if path == "api/chat":
            msg = json.loads(body.decode())["messages"][0]
            panel = Image.open(io.BytesIO(base64.b64decode(msg["images"][-1]))).convert("RGB")
            oracle.calls += 1
            if msg["content"].startswith("Panel images A"):
                ans = str(oracle.pair_answer(panel.crop((0, 20, 224, 244)), panel.crop((224, 20, 448, 244))))
            else:
                ans = oracle.answer_tile(panel.crop((0, 20, 224, 244)))
            return 200, json.dumps({"message": {"content": ans}}).encode()
        return 404, b"{}"
    return transport


def rl_a_answers(oracle, sheets_dir, out_dir, domain, labeller):
    """RL-A (the external labeller, lab side) answering every pool sheet in one
    reply per sheet, recorded through rl.answers_from_text."""
    from PIL import Image
    from weed_optimizer_framework.tools.funnel import rl as RLM
    n = 0
    for p in sorted(sheets_dir.glob("sheet_*.json")):
        doc = json.loads(p.read_text())
        with Image.open(sheets_dir / doc["image"]["file"]) as im0:
            im = im0.convert("RGB")
        lines = []
        for it in sorted(doc["items"], key=lambda x: x["position"]):
            x, y, w, h = it["tiles"]["crop"]
            lines.append("%d: %s" % (it["position"], oracle.answer_tile(im.crop((x, y, x + w, y + h)))))
        RLM.answers_from_text(p, "\n".join(lines), "RL-A", labeller, out_dir=out_dir, domain=domain)
        n += 1
    return n


def kt7_transport(domain, colors, per_taxon):
    """A fake iNaturalist: per taxon, per_taxon research-grade observations
    with one CC photo each; a target taxon's photo is painted in its class
    colour, any other taxon's in the other class's."""
    import numpy as np
    from PIL import Image
    targets = {t["taxon"]: int(t["id"]) for t in domain.targets}
    other = int(domain.other["id"])
    photos, obs = {}, {}
    oid = 1000
    for taxon in domain.raw["known_truth"]["kt7"]["taxa"]:
        cls = targets.get(taxon, other)
        res = []
        for _j in range(per_taxon):
            oid += 1
            pid = 50000 + oid
            rng = np.random.default_rng(oid)
            small = rng.integers(30, 225, size=(6, 8, 3), dtype=np.uint8)
            arr = np.array(Image.fromarray(small).resize((96, 72), Image.BILINEAR), dtype=np.int16)
            arr[18:54, 24:72] = np.array(colors[cls], dtype=np.int16) + rng.integers(-10, 11, size=(36, 48, 3))
            buf = io.BytesIO()
            Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(buf, "PNG")
            photos["https://static.example.org/photos/%d/large.png" % pid] = buf.getvalue()
            res.append({"id": oid, "quality_grade": "research", "taxon": {"name": taxon},
                        "observed_on": "2025-06-01", "place_guess": "a field",
                        "photos": [{"id": pid, "license_code": "cc-by",
                                    "url": "https://static.example.org/photos/%d/square.png" % pid}]})
        obs[taxon] = res

    def transport(url, params=None, headers=None):
        if url in photos:
            return 200, photos[url], {}
        params = dict(params or {})
        if params.get("taxon_name") in obs:
            page = int(params.get("page", "1"))
            body = {"total_results": len(obs[params["taxon_name"]]),
                    "results": obs[params["taxon_name"]] if page == 1 else []}
            return 200, json.dumps(body).encode(), {}
        return 404, b"{}", {}
    return transport


# ====================================================================== the world
SP8 = "cottonweed_sp8"
N_VETO = 80
N_JUDGE = 40
N_GUARDED = 30
N_CARD = 34
NOT_RECOVERABLE = "rf_uav-qnoms__uav-wqshy"      # domains/weed.json sources.not_recoverable
KT7_PER_TAXON = 18


def plant(ctx):
    """funnel_world's plant hook (see the module docstring)."""
    import itertools
    import numpy as np
    from PIL import Image
    import funnel_world as FW
    from weed_optimizer_framework.tools import cwd12_species as S
    from weed_optimizer_framework.tools.funnel import leak as LK
    from weed_optimizer_framework.tools.inc import verify as V
    C = ctx.C
    P = ctx.painter
    bits = lambda a, b: bin(int(a) ^ int(b)).count("1")    # noqa: E731
    evals = [r for s in ("dev", "test") for r in ctx.rows[s]]
    eval_h = [C.dhash(r["image"]) for r in evals]
    core = ctx.core
    core_h = {r["key"]: C.dhash(r["image"]) for r in core}
    planted = {"negatives": [], "h6_copy": None, "copies": {}, "veto": [], "judge": [], "card": [], "guarded": []}

    # (1) the reference-copy sources: one class of >= 20 boxes each, in pictures no other source copies
    used = {4, 33, 8, 24, 5}                       # funnel_world's own copies (holdout, three_season, cwp10)
    classes_of = [sorted({b[0] for b in ctx.boxes_of[r["image"]]}) for r in core]
    pics = {c: {i for i, cl in enumerate(classes_of) if c in cl} for c in range(12)}
    # cottonweed_sp8's files hold trainer slot ids 0-7 (its eight species), the holdout's the class ids
    sp8_slot = {V.inc_id_of_slot(slot): slot for slot in range(8)}
    best = None
    for a, b, c in itertools.permutations(range(12), 3):
        if a > 3 or c not in sp8_slot:
            continue                                # the holdout's old join is wrong on ids 0-3 only
        sa, sb, sc = pics[a] - (used - {4, 33}), pics[b] - (used - {8, 24}), pics[c] - used
        if sa & sb or sa & sc or sb & sc:
            continue
        score = min(len(sa), len(sb), len(sc))
        if best is None or score > best[0]:
            best = (score, (a, sa), (b, sb), (c, sc))
    assert best and best[0] >= 20, "no class assignment gives every copy source 20 boxes: %s" % (best,)
    for slug, (cls, idx) in zip((FW.HOLDOUT, FW.THREE, SP8), best[1:]):
        planted["copies"][slug] = {"class": cls, "pictures": len(idx)}
        dd = ctx.datasets / slug / "train"
        for i in sorted(idx):
            r = core[i]
            stem = pathlib.Path(r["image"]).stem
            if (dd / "images" / (stem + ".jpg")).exists():
                continue
            (dd / "images").mkdir(parents=True, exist_ok=True)
            shutil.copyfile(r["image"], dd / "images" / (stem + ".jpg"))
            boxes = ctx.boxes_of[r["image"]]
            if slug == SP8:
                boxes = [(sp8_slot[b[0]],) + tuple(b[1:]) for b in boxes if b[0] in sp8_slot]
            FW._yolo(dd / "labels" / (stem + ".txt"), boxes)
    ctx.registry_extra[SP8] = {"local_path": str(ctx.datasets / SP8), "annotation": "bbox",
                               "class_names": list(S.CWD12_SPECIES)}

    def own_size(boxes):
        """Boxes 3 and 5 px smaller than funnel_world's, so the geometry match of the card check
        (2 px tolerance) never pairs a planted image with a filler image of the same source."""
        return [(c, cx, cy, 0.2, 0.28) for c, cx, cy, _w, _h in boxes]

    # (2) vetoed images of the authoritative source (G2v, R-V)
    wc = FW.WEEDCROP
    for i in range(N_VETO):
        stem = "wc_vet%02d" % i
        ctx.add(wc, stem, own_size(P.layout3([0, 8, 12])),
                [ctx.src_id(wc, "Waterhemp"), ctx.src_id(wc, "Blackbean"), ctx.src_id(wc, "Kochia")])
        planted["veto"].append(stem)

    # (2b) numeric-source images whose id-3 box is painted Purslane (a no-information conflict stratum,
    # recovery policy R-J)
    for i in range(N_JUDGE):
        stem = "mh_pur%02d" % i
        ctx.add(FW.MH, stem, own_size(P.layout3([2, 12, 12])), [3, 1, 4])
        planted["judge"].append(stem)

    # (2d) a card-resolved numeric class with at least G3_MIN_BOXES boxes (the numeric source's id 5, whose
    # card taxon is the genus-level Morning glory class): the H3a stratum. The authoritative sources' named
    # crops (Blackbean) resolve to a taxon, so they are never class-level units.
    for i in range(N_CARD):
        stem = "mh_card%02d" % i
        ctx.add(FW.MH, stem, own_size(P.layout3([1, 1, 1])), [5, 5, 5])
        planted["card"].append(stem)

    # (2c) vetoed images like (2) in the licensed greenhouse source, their blocking box a Blackbean painted
    # Morning glory (the contract's attractor), and the Purslane of (2b) in a source the config marks not
    # recoverable: the gates of their strata pass, so only the guards keep them out of every recovered pool
    # (the greenhouse source is quarantined as a whole by the dev copy of (4))
    gh = FW.GREENHOUSE
    for i in range(N_GUARDED):
        stem = "gh_vet%02d" % i
        ctx.add(gh, stem, own_size(P.layout3([0, 1, 12])),
                [ctx.src_id(gh, "Waterhemp"), ctx.src_id(gh, "Blackbean"), ctx.src_id(gh, "Redroot Pigweed")])
        planted["guarded"].append((gh, stem))
    d_nr = ctx.datasets / NOT_RECOVERABLE
    for i in range(N_GUARDED):
        stem = "nr_pur%02d" % i
        img = d_nr / "images" / (stem + ".png")
        bx = own_size(P.layout3([2, 12, 12]))
        P.paint(img, bx, texture=False, fmt="png")
        FW._yolo(d_nr / "labels" / (stem + ".txt"), [(0,) + tuple(b[1:]) for b in bx])
        planted["guarded"].append((NOT_RECOVERABLE, stem))
    ctx.registry_extra[NOT_RECOVERABLE] = {"local_path": str(d_nr), "annotation": "bbox", "class_names": ["0"]}

    # (3) the copy detector's calibration negatives: 7-10 dHash bits from a train_core picture, recoloured
    # to one luma-preserving hue (another photograph as far as the descriptor is concerned)
    rng = np.random.default_rng(11)
    for i in range(6):
        r = core[12 + 2 * i]
        with Image.open(r["image"]) as im:
            base = im.convert("L")
        W, H = base.size
        img = None
        for _attempt in range(4000):
            L = np.asarray(base, dtype=np.float64).copy()
            for _k in range(int(rng.integers(0, 4))):
                x0, y0 = int(rng.integers(0, W - 16)), int(rng.integers(0, H - 12))
                L[y0:y0 + 12, x0:x0 + 16] += float(rng.integers(-70, 70))
            rgb = np.clip(L, 0, 255)[..., None] * np.array([0.45, 1.3, 0.75])[None, None, :]
            x0, x1 = int(round(0.44 * W)), int(round(0.56 * W))
            y0, y1 = int(round(0.05 * H)), int(round(0.19 * H))
            rgb[y0:y1, x0:x1] = np.array(FW.COLORS[12], dtype=np.float64)
            cand = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
            hv = LK.dhash_variants(cand)
            d = bits(hv["id"], core_h[r["key"]])
            if (7 <= d <= 10 and min(bits(hv[v], core_h[r["key"]]) for v in LK.VARIANTS) > 6
                    and min(bits(hv["id"], e) for e in eval_h) > 6
                    and min(bits(hv["id"], h2) for k2, h2 in core_h.items() if k2 != r["key"]) > 6):
                img = cand
                break
        assert img is not None, "no calibration negative near %s" % r["key"]
        stem = "mh_neg_%d" % i
        p = ctx.datasets / FW.MH / "images" / (stem + ".png")
        img.save(p)
        FW._yolo(ctx.datasets / FW.MH / "labels" / (stem + ".txt"), [(2, 0.5, 0.12, 0.12, 0.14)])
        P.hashes.append(C.dhash(p))
        planted["negatives"].append({"stem": stem, "train_core_key": r["key"], "bits": d})

    # (4) a flipped copy of a dev picture in the licensed source of (2c): its plain dHash is far from the
    # dev picture (Step 1 kept it); the detector's flip variant finds it
    ev = ctx.rows["dev"][1]
    with Image.open(ev["image"]) as im:
        cp = im.convert("RGB").transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    stem = "gh_devflip"
    p = ctx.datasets / gh / "images" / (stem + ".png")
    cp.save(p)
    h = C.dhash(p)
    assert min(bits(h, e) for e in eval_h) > 6 and min(bits(h, c) for c in core_h.values()) > 6
    FW._yolo(ctx.datasets / gh / "labels" / (stem + ".txt"),
             [(ctx.src_id(gh, "Blackbean"), round(1 - b[1], 4)) + tuple(b[2:]) for b in ctx.boxes_of[ev["image"]]])
    P.hashes.append(h)
    planted["h6_copy"] = {"stem": stem, "eval_key": ev["key"]}
    ctx.world.pipeline_planted = planted


def voc(fname, W, H, objs):
    o = "".join("<object><name>%s</name><bndbox><xmin>%g</xmin><ymin>%g</ymin><xmax>%g</xmax><ymax>%g</ymax>"
                "</bndbox></object>" % tuple(x) for x in objs)
    return ("<annotation><filename>%s</filename><size><width>%d</width><height>%d</height><depth>3</depth></size>%s"
            "</annotation>" % (fname, W, H, o))


def write_cards(w, fd, dom, FW, C):
    """The fetched-card layout (what fetch --what cards writes on the lab,
    runner §4.5): cards/index.json and the upstream annotations of the card
    resolvers with geometry, built from the pool's own label files (MH-Weed16
    as a VOC + YOLO archive with the card's names; the two authoritative
    sources as YOLO folders with classes.txt)."""
    import zipfile
    meta = [json.loads(ln) for ln in open(w.step1 / "pool_meta.jsonl")]
    cards = {}
    z = fd / "cards" / FW.MH / "annotations.zip"
    z.parent.mkdir(parents=True)
    mh_table = dom.raw["sources"]["card_resolvers"][FW.MH]["class_table"]
    with zipfile.ZipFile(z, "w") as zf:
        for m in (m for m in meta if m["source"] == FW.MH):
            stem = m["key"].split("__")[-1]
            W, H = m["W"], m["H"]
            objs, lines = [], []
            for b, (_c, cx, cy, bw, bh) in enumerate(m["boxes"]):
                sid = int(m["src"][b][0])
                objs.append((mh_table[str(sid)]["name"], (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W,
                             (cy + bh / 2) * H))
                lines.append("%d %.6f %.6f %.6f %.6f\n" % (sid, cx, cy, bw, bh))
            zf.writestr("a/PASCAL_VOC/%s.xml" % stem, voc("VID_1.mp4_%d.png" % len(objs), W, H, objs))
            zf.writestr("a/YOLO_darknet/%s.txt" % stem, "".join(lines))
    cards[FW.MH] = [{"what": "annotations", "file": "%s/annotations.zip" % FW.MH, "sha256": C.sha256_file(z)}]
    for slug in (FW.WEEDCROP, FW.GREENHOUSE):
        ents = []
        base = fd / "cards" / slug / "annotations" / "Folder_A"
        base.mkdir(parents=True)
        (base / "classes.txt").write_text("\n".join(FW.SOURCE_NAMES[slug]) + "\n")
        ents.append({"what": "annotations", "file": "%s/annotations/Folder_A/classes.txt" % slug,
                     "sha256": C.sha256_file(base / "classes.txt")})
        for m in (m for m in meta if m["source"] == slug):
            stem = m["key"].split("__")[-1]
            p = base / (stem + ".txt")
            p.write_text("".join("%d %.6f %.6f %.6f %.6f\n" % (int(m["src"][b][0]), cx, cy, bw, bh)
                                 for b, (_c, cx, cy, bw, bh) in enumerate(m["boxes"])))
            ents.append({"what": "annotations", "file": "%s/annotations/Folder_A/%s.txt" % (slug, stem),
                         "sha256": C.sha256_file(p)})
        cards[slug] = ents
    (fd / "cards" / "index.json").write_text(json.dumps({"format": "funnel-cards/1", "cards": cards}))


# ====================================================================== the run
def main():
    missing = []
    for mod in ("numpy", "PIL", "sklearn", "joblib"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        skip("the pipeline", "%s missing" % ", ".join(missing))
        finish()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="funnel_pipeline_"))
    os.environ["INC_DIR"] = str(tmp / "inc")
    os.environ["REPO"] = str(tmp / "repo")
    os.environ["INC_SCORER_TESTING"] = "1"                 # the realloop builds here are testing ones
    for k in ("FUNNEL_CONTRACT", "FUNNEL_DOMAINS_DIR", "FUNNEL_OLLAMA_ENDPOINT", "SLURM_JOB_ID"):
        os.environ.pop(k, None)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(TESTS))

    class _NoNet(socket.socket):
        def __init__(self, *a, **k):
            raise AssertionError("the pipeline test touched the network")
    socket.socket = _NoNet

    import funnel_world as FW
    from weed_optimizer_framework.tools.inc import common as C
    from weed_optimizer_framework.tools.funnel import __main__ as CLI
    from weed_optimizer_framework.tools.funnel import domain as D
    from weed_optimizer_framework.tools.funnel import embed as E
    from weed_optimizer_framework.tools.funnel import fetch as F
    from weed_optimizer_framework.tools.funnel import rl as RLM
    from weed_optimizer_framework.tools.funnel.adapters import inc_step1 as A

    print("the world")
    w = FW.build_world(tmp, n_images=int(os.environ.get("FUNNEL_PIPELINE_NIMG", "240")), plant=plant,
                       core_frames=9)
    print("  built in %.1fs: %s" % (time.time() - T0, json.dumps(w.pipeline_planted)[:300]))
    fd = w.funnel_dir
    r1 = w.inc_dir / "step1_r1"
    prereg = fd / "prereg_v1.json"
    pj = json.loads(prereg.read_text())
    for g, v in pj["sampling"]["groups"].items():
        if v[1] is not None:
            v[1] = 0
    prereg.write_text(json.dumps(pj, indent=1, ensure_ascii=False) + "\n")
    dom = D.load("weed")
    FakeDino.COLORS = FW.COLORS
    E.Dinov2Embedder = FakeDino
    E.LazyEmbedder = FakeDino
    A.text_encoder = lambda: FakeText(dom)
    A.bioclip_embedder = lambda: Kt7ProbeEmbedder(FW)
    gbif, kt7 = FW.replay_transport(), kt7_transport(dom, FW.COLORS, KT7_PER_TAXON)

    def web(url, params=None, headers=None):
        """The lab's network for fetch: the recorded GBIF answers and the fake iNaturalist."""
        return gbif(url, dict(params or {})) if "gbif.org" in url else kt7(url, params, headers)
    F.default_transport = web
    oracle = Oracle(dom, FW.COLORS)
    RLM._urllib_transport = ollama_transport(oracle, dom.raw["reference_labeller"]["backends"]["RL-B"]["model"])

    def run(*argv, out=None):
        t = time.time()
        rc = CLI.main(list(argv) + ["--prereg", str(prereg), "--testing", "--quiet", "--out", str(out or fd)])
        print("  [%s: exit %d, %.1fs]" % (" ".join(argv[:3]), rc, time.time() - t))
        return rc

    steps = [("fetch", "--what", "taxonomy", "--names-from", str(w.step1 / "pool_summary.json")), ("census",),
             ("realloop_v1",), ("fetch", "--what", "kt7"), ("arrival",), ("leak",), ("cards",),
             ("map", "--part", "geometry"), ("embed-judges",), ("qualify",), ("map", "--part", "relation"),
             ("draw",), ("sheets",), ("rl-b", "--endpoint", "http://127.0.0.1:9"), ("rl-a",), ("ingest",),
             ("qualify", "--rl"), ("da",), ("estimate",),
             ("recover", "--audit", str(fd / "audit_v1.json"), "--maps", str(fd / "class_maps.json"),
              "--policy", "R-A,R-C,R-T,R-V,R-J"), ("realloop_v2",),
             ("recover", "--arms", "--realloop", "realloop_v2", "--base", str(w.step1 / "base_B.jsonl"))]
    stop_after = os.environ.get("FUNNEL_PIPELINE_STOP")

    def in_process(name):
        """The steps that are not funnel verbs; each one's refusal is a failure of this test."""
        from weed_optimizer_framework.tools.inc import driver as DRV
        from weed_optimizer_framework.tools.inc import realloop as RL
        if name == "realloop_v1":
            RL.build("realloop_v1", base=w.step1 / "base_B.jsonl", n_verified=2, size=2, replay_mode="full",
                     recipes="full", testing=True, init=False, quiet=True)
        elif name == "realloop_v2":
            base_n = len(C.read_manifest(w.step1 / "base_B.jsonl"))
            m = int(0.05 * base_n) + 2                   # above the floor of 5 % of B (contract §9.1)
            print("  realloop_v2: |B| %d, M %d" % (base_n, m))
            RL.build("realloop_v2", base=w.step1 / "base_B.jsonl", size=m, replay_mode="full", recipes="full",
                     gate_flips_mode="net", increment_sources="recovered", step1_overlay=r1, testing=True,
                     backend=DRV.FakeBackend(), quiet=True)   # driver init writes exp.json; nothing is submitted
        elif name == "cards":
            write_cards(w, fd, dom, FW, C)
        elif name == "arrival":
            F.check_manifest(fd)                          # the cluster's check of what the lab fetched (§6.3)
        elif name == "rl-a":
            model = dom.raw["reference_labeller"]["backends"]["RL-A"]["model"]
            rl_a_answers(oracle, fd / "sheets_v1", fd / "rl_answers" / "RL-A", dom, "RL-A:%s" % model)
        elif name == "da":
            write_da(fd, dom, prereg, D)
        else:
            return False
        return True
    for st in steps:
        name = st[0]
        try:
            done = in_process(name)
        except Exception as e:                           # noqa: BLE001 - reported as this step's failure
            check("%s completes" % name, False, "%s: %s" % (type(e).__name__, e))
            finish()
        if done:
            continue
        rc = run(*st, out=r1 if name == "recover" else None)
        check("%s exits 0" % " ".join(st[:3]), rc == 0, rc)
        if rc != 0:
            finish()
        if stop_after and name == stop_after:
            break
    report(w, fd, r1)
    verify(w, fd, r1, dom, prereg, FW, C)
    reruns(run, w, fd, r1, C)
    regressions(fd, dom)
    finish()


def write_da(fd, dom, prereg, D):
    """prospective_da.json (a synthetic fixture: on the platform the lab writes
    it from a validated devil's-advocate reply before any estimate exists)."""
    from weed_optimizer_framework.tools import funnel as FUN
    from weed_optimizer_framework.tools.funnel import ledger as LG
    led = LG.load(fd / "funnel_ledger.json")
    rec = [st["id"] for st in led["stages"] if st.get("recoverable") is True]
    fc = {sid: round(0.4 / (len(rec) - 1), 12) for sid in rec}
    top = next(s["id"] for s in led["stages"] if s.get("role") == "image_rule")
    fc[top] = 0.6
    pre = D.load_prereg(prereg)
    doc = FUN.header("prospective_da", dom, pre, {}, testing=True)
    doc.update({"claim_ids": ["C1"], "digest_sha256": "0" * 64, "blind_check": {"markers": [], "found": []},
                "model": {"role": "adversary", "resolved": "vllm:glm-4.7-flash", "family": "glm",
                          "planner_resolved": "ollama:qwen3.8:27b", "planner_family": "qwen", "same_family": False},
                "reply": {}, "validation": {"ok": True}, "stage_forecast": fc,
                "committed_utc": "2026-09-28T00:00:00Z", "synthetic_fixture": True})
    FUN.write_json_atomic(fd / "prospective_da.json", doc)


# ====================================================================== the checks
def _jsonl(path):
    return [json.loads(ln) for ln in open(path) if ln.strip()]


def _csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


class Truth(object):
    """The world's planted truth: the class each pool box was painted as,
    read from the source picture (lossless), never from a verdict."""

    def __init__(self, step1, colors):
        import numpy as np
        self.np = np
        self.colors = np.array(colors, dtype=np.float64)
        self.pool = {r["key"]: r for r in _jsonl(step1 / "pool.jsonl")}
        self.meta = {m["key"]: m for m in _jsonl(step1 / "pool_meta.jsonl")}
        self._arr = {}

    def array(self, path):
        from PIL import Image
        if path not in self._arr:
            with Image.open(path) as im:
                self._arr[path] = self.np.asarray(im.convert("RGB"), dtype=self.np.float64)
        return self._arr[path]

    def at(self, path, cx, cy):
        a = self.array(str(path))
        H, W = a.shape[:2]
        x, y = int(cx * W), int(cy * H)
        c = a[max(0, y - 2):y + 3, max(0, x - 2):x + 3].reshape(-1, 3).mean(0)
        return int(self.np.argmin(((self.colors - c) ** 2).sum(1)))

    def box(self, key, b):
        m = self.meta[key]
        bx = m["boxes"][int(b)]
        return self.at(self.pool[key]["image"], bx[1], bx[2])

    def unit(self, unit_id):
        key, _, b = unit_id[2:].partition("#")
        return self.box(key, int(b))


def verify(w, fd, r1, dom, prereg, FW, C):
    import numpy as np
    from PIL import Image
    from weed_optimizer_framework.tools.funnel import domain as D
    from weed_optimizer_framework.tools.funnel import draw as DR
    from weed_optimizer_framework.tools.funnel import ledger as LG
    from weed_optimizer_framework.tools.inc import pilot as PL
    from weed_optimizer_framework.tools.inc_autopilot import diagnose as DG
    from weed_optimizer_framework.tools.inc_autopilot import evidence as EV
    from weed_optimizer_framework.tools.inc_autopilot import outcome as OC
    pre = D.load_prereg(prereg)
    pkg = ROOT / "weed_optimizer_framework" / "tools"
    contract_sha = C.sha256_file(w.repo / "docs" / "FUNNEL_AUDIT.md")
    truth = Truth(w.step1, FW.COLORS)
    targets = set(range(len(dom.targets)))
    planted = w.pipeline_planted
    sha = C.sha256_file

    print("the freshness chain (runner §7.3): headers, recorded inputs and code")
    arts = ["census_v1.json", "name_status_v2.json", "funnel_ledger.json", "leak_v2.json", "kt7/kt7_index.json",
            "relation_geometry_v1.json", "class_maps.json", "judges/index.json", "judge_qualification.json",
            "relation_audit_v1.json", "frames_v1.json", "sheets_v1/index.json", "sheets_v1_cluster/index.json",
            "gold_v1.json", "rl_qualification.json", "audit_v1.json"]
    docs = {a: json.loads((fd / a).read_text()) for a in arts}
    for a in ("recovery.json", "domain_dev.json", "arms/arms.json"):
        docs["step1_r1/" + a] = json.loads((r1 / a).read_text())
    bad_hdr, stale, bad_code, n_inputs = [], [], [], 0
    for a, d in sorted(docs.items()):
        if ((d.get("prereg") or {}).get("core_sha256") != pre.core_sha256
                or (d.get("contract") or {}).get("sha256") != contract_sha
                or (d.get("domain_config") or {}).get("sha256") != dom.sha256):
            bad_hdr.append(a)
        for name, rec in sorted((d.get("inputs") or {}).items()):
            if not isinstance(rec, dict) or not rec.get("path") or not rec.get("sha256"):
                continue
            if os.path.abspath(rec["path"]) == os.path.abspath(str(prereg)):
                continue                        # the sample lock amends the prereg; its core is compared above
            n_inputs += 1
            if not os.path.isfile(rec["path"]) or sha(rec["path"]) != rec["sha256"]:
                stale.append("%s: %s" % (a, name))
        for rel, h in sorted((d.get("code") or {}).items()):
            p = pkg / rel
            if p.is_file() and sha(p) != h:
                bad_code.append("%s: %s" % (a, rel))
    check("every artifact records this prereg's core, the contract and the domain config", not bad_hdr, bad_hdr)
    check("every recorded input (%d over %d artifacts) still hashes as recorded" % (n_inputs, len(docs)),
          n_inputs > 100 and not stale, stale[:10])
    check("every artifact's code record is the code that ran", not bad_code, bad_code[:10])
    links = [("frames_v1.json", "census_v1", fd / "census_v1.json"),
             ("frames_v1.json", "name_status_v2", fd / "name_status_v2.json"),
             ("frames_v1.json", "judge_qualification", fd / "judge_qualification.json"),
             ("frames_v1.json", "relation_geometry_v1", fd / "relation_geometry_v1.json"),
             ("sheets_v1/index.json", "sample", fd / "sample_v1.csv"),
             ("gold_v1.json", "key", fd / "sheets_v1_key" / "key.jsonl"),
             ("rl_qualification.json", "gold", fd / "gold_v1.csv"),
             ("audit_v1.json", "rl_qualification", fd / "rl_qualification.json"),
             ("audit_v1.json", "prospective_da", fd / "prospective_da.json"),
             ("audit_v1.json", "sample_v1", fd / "sample_v1.csv"),
             ("audit_v1.json", "leak_v2", fd / "leak_v2.json"),
             ("step1_r1/recovery.json", "audit", fd / "audit_v1.json"),
             ("step1_r1/recovery.json", "leak", fd / "leak_v2.json"),
             ("step1_r1/arms/arms.json", "recovery", r1 / "recovery.json"),
             ("step1_r1/arms/arms.json", "exp", w.inc_dir / "realloop_v2" / "exp.json")]
    missing = [(a, n) for a, n, p in links if (docs[a].get("inputs") or {}).get(n, {}).get("sha256") != sha(p)]
    check("each consumer records its producer's file (%d links, census -> ... -> arms)" % len(links), not missing,
          missing)
    lock = pre.sample_lock or {}
    check("the sample lock follows the prereg's contract amendment A2 (as A3) and names the sample, key, frames and "
          "name status", [(a.get("id"), a.get("kind")) for a in pre.amendments] == [("A2", "amendment"), ("A3", "sample_lock")]
          and lock.get("sample_sha256") == sha(fd / "sample_v1.csv")
          and lock.get("key_sha256") == sha(fd / "sample_v1_key.jsonl")
          and lock.get("frames_sha256") == sha(fd / "frames_v1.json")
          and lock.get("name_status_v2_sha256") == sha(fd / "name_status_v2.json"), lock)
    check("the drawn sample loads against the lock", len(DR.load_sample(fd / "sample_v1.csv", pre)) > 0)
    check("the audit's ledger fingerprint is the funnel ledger's",
          docs["audit_v1.json"]["ledger_fingerprint"] == docs["funnel_ledger.json"]["fingerprint"])

    print("F3 census, F4 leak and maps")
    cen = docs["census_v1.json"]
    check("the census reconciles with the Step 1 summaries", cen["reconciliation"]["ok"],
          [c for c in cen["reconciliation"]["checks"] if not c["ok"]])
    check("the funnel ledger validates", not LG.validate(docs["funnel_ledger.json"]),
          LG.validate(docs["funnel_ledger.json"])[:5])
    # the prereg's amendment A2 (contract §14) requires copy detector version 2: the leak verb wrote leak_v2.json
    lk = docs["leak_v2.json"]
    v1n = ((lk["readings"]["v1"].get("calibration") or {}).get("negatives") or {}).get("pairs_7_10") or {}
    check("the copy detector (version 2, per image) passed its calibration; version 1's per-pair calibration, "
          "rebuilt, holds the planted negatives (7-10 bits, provenance-disjoint)",
          lk["detector_version"] == 2 and not (fd / "leak_v1.json").exists() and lk["calibration"]["ok"]
          and lk["calibration"]["negatives"]["hard"]["n"] > 0 and v1n.get("n", 0) >= len(planted["negatives"])
          and lk["calibration"]["cos_threshold"] >= lk["calibration"]["floor"], lk["calibration"]["why"])
    gh_copy = "%s__%s" % (FW.GREENHOUSE, planted["h6_copy"]["stem"])
    listed = [(c["key"], c["eval_key"]) for c in lk["scans"]["source:%s" % FW.GREENHOUSE]["listed"]]
    check("H6(a): the flipped dev copy is found (dHash variants) and its source is quarantined as a whole",
          (gh_copy, planted["h6_copy"]["eval_key"]) in listed and lk["h6a"]["quarantine"] == [FW.GREENHOUSE]
          and lk["scans"]["source:%s" % FW.GREENHOUSE]["dhash_hits"] >= 1, (listed, lk["h6a"]["quarantine"]))
    check("H6(b): base B and realloop_v1's increments hold no copy", lk["h6b"]["incident"] is False
          and lk["h6b"]["cleared"] is True, lk["h6b"])
    ra = docs["relation_audit_v1.json"]
    check("H0(b): the relation audit maps, flags and keeps as the planted copies say", ra["h0b"]["pass"] is True,
          ra["h0b"]["checks"])

    print("F6 draw, F7 sheets and labels")
    key_rows = {r["item_id"]: r for r in _jsonl(fd / "sheets_v1_key" / "key.jsonl")}
    pool_sheets = {p.stem for p in (fd / "sheets_v1").glob("sheet_*.json")}
    wrong_dir = [k for k, r in key_rows.items() if (r["group"] == "G5" or str(r.get("stratum", "")).startswith(
        "pair_sentinel")) == (r["sheet_id"] in pool_sheets)]
    check("G5 pairs and pair sentinels are on cluster sheets only, every pool item on a pool sheet", not wrong_dir,
          wrong_dir[:5])
    slugs = sorted({r["source"] for r in truth.pool.values()} | {FW.HOLDOUT, FW.THREE, SP8})
    leaks = []
    for p in sorted((fd / "sheets_v1").glob("*.json")):
        text = p.read_text()
        for needle in slugs + ["b:", "G1/", "G2/", "G4/", "stratum", "verdict"]:
            if needle in text:
                leaks.append((p.name, needle))
    check("blinding: no source slug, unit id, stratum or verdict in any pool sheet or board file", not leaks,
          leaks[:5])
    ans_a = {p.stem for p in (fd / "rl_answers" / "RL-A").glob("sheet_*.json")}
    check("RL-A (outside the cluster) answered pool sheets only (DEC-2)", ans_a and ans_a <= pool_sheets,
          sorted(ans_a - pool_sheets)[:5])
    rq = docs["rl_qualification.json"]
    prim = rq["primary"]["H0"]
    check("the machine RL qualifies at species level in H0's scope from the sentinels alone",
          prim.get("qualified") is True and prim.get("level") == "species", prim)
    gold = _csv(fd / "gold_v1.csv")
    wrong_sent = [g["item_id"] for g in gold if g["is_sentinel"] in ("1", "True", "true")
                  and g["group"] == "sentinel" and g["correct_species"] != "1"]
    check("the stub labeller answered every sentinel right at species level (it reads the painted colour)",
          not wrong_sent, wrong_sent[:5])

    print("F8 estimates against the planted truth")
    au = docs["audit_v1.json"]
    check("the audit is valid and does not stop the campaign", au["valid"] is True and au["stop"] is None,
          (au["valid"], au["stop"]))
    h0 = au["hypotheses"]["H0"]
    share = (h0["parts"]["a"] or {}).get("planted_share")
    key_planted = [r["planted"] for r in _jsonl(fd / "sample_v1_key.jsonl") if "planted" in r][0]
    check("H0 supported: G0's interval covers the planted share, which the key holds",
          h0["verdict"] == "supported" and share == key_planted["share"]
          and h0["interval"][0] <= share <= h0["interval"][1], (h0["verdict"], share, h0["interval"]))
    frame_rows = {}
    for g in ("G1", "G2", "G2v", "G2a", "G4"):
        for r in _csv(fd / "frames_v1" / ("%s.csv" % g)):
            frame_rows.setdefault(r["stratum"], []).append(r)
    covered, missed, n_rec = 0, [], 0
    for rec in au["strata"]:
        if rec["group"] not in ("G1", "G2", "G2v", "G2a", "G4") or not rec["n_labelled"]:
            continue
        ev_ = rec["event"]
        if ev_ not in ("target", "label", "pred"):
            continue
        units = frame_rows[rec["stratum"]]
        ys = []
        for u in units:
            t = truth.unit(u["unit_id"])
            if ev_ == "target":
                ys.append(int(t in targets))
            else:
                # estimate.py's "label" and "pred": the answer is that TARGET class and the box is valid
                # (a non-target label or prediction is never right: nothing would be relabelled)
                want = u["label"] if ev_ == "label" else u["pred"]
                ys.append(int(want in dom.target_names and t == dom.class_id(want)))
        tshare = sum(ys) / float(len(ys))
        n_rec += 1
        lo, hi = rec["interval"]
        if lo - 1e-9 <= tshare <= hi + 1e-9:
            covered += 1
        else:
            missed.append((rec["stratum"], ev_, round(tshare, 4), [round(lo, 4), round(hi, 4)]))
    check("every labelled box stratum's interval covers its frame's true share (%d of %d estimates)"
          % (covered, n_rec), n_rec >= 20 and not missed, missed[:8])
    need = {"G2": "label", "G2v": "label", "G1": "pred", "G4": "pred"}
    have = collections.defaultdict(set)
    for rec in au["strata"]:
        have[rec["stratum"]].add(rec["event"])
    lacking = [s for s in have if s.split("/", 1)[0] in need and need[s.split("/", 1)[0]] not in have[s]]
    check("the audit holds, per stratum, the event each recovery gate reads (label, pred)", not lacking, lacking)
    from weed_optimizer_framework.tools.funnel import estimate as ES
    from weed_optimizer_framework.tools.funnel import qualify as Q
    stub = type("HypOf", (), {"_card_sources": set(Q._card_resolved_sources(dom))})()
    hs = rq.get("hypothesis_strata") or {}
    off = [(h, sid, ES.Context.hyp_of(stub, sid)) for h in ("H1", "H2a", "H2b", "H3a", "H3b", "H4")
           for sid in hs.get(h) or [] if ES.Context.hyp_of(stub, sid) != h]
    check("estimate answers each stratum with the hypothesis qualify picked its primary for (%d strata)"
          % sum(len(v or []) for v in hs.values()), hs.get("H3a") and not off, off[:5])
    oc = OC.h11(json.loads((fd / "prospective_da.json").read_text()),
                dict(au, hypotheses={k: v for k, v in au["hypotheses"].items() if k != "H11"}),
                [st["id"] for st in docs["funnel_ledger.json"]["stages"] if st.get("recoverable") is True])
    check("H11: the autopilot's scorer (outcome.h11) and the estimator agree on the same files",
          oc["verdict"] == au["hypotheses"]["H11"]["verdict"], (oc, au["hypotheses"]["H11"]["why"]))
    ev = EV.load_dir(w.inc_dir, "realloop_v1", exps=["realloop_v1"])
    d18 = [d for d in DG.detect(ev) if d["id"] == "D18"][0]
    rec_doc = docs["step1_r1/recovery.json"]
    rows = _jsonl(r1 / "recovered_pool.jsonl")
    got_policies = {r["policy"] for r in rows}
    d18_policies = set((d18["detail"].get("policy") or "").split(",")) - {""}
    check("D18 reads this audit (valid, this Step 1's fingerprint) and every policy it proposes recovered rows",
          d18["fired"] and d18_policies and d18_policies <= got_policies, (d18["summary"], got_policies))

    from weed_optimizer_framework.tools.inc_autopilot import remote as RM
    summ = RM.funnel_summary()
    arts = summ["decision"]["artifacts"]
    sa = arts.get("funnel/audit_v1.json") or {}
    shipped = {n for n, _rel in RM.FUNNEL_SHIP}
    check("remote funnel summary ships every aggregate, and the audit's d18_inputs, stage ranking and verdicts "
          "reach the lab unredacted",
          summ["ok"] and not summ["missing"] and shipped <= set(arts)
          and sa.get("d18_inputs") == au["d18_inputs"] and sa.get("stage_ranking") == au["stage_ranking"]
          and {k: v["verdict"] for k, v in sa.get("hypotheses", {}).items()}
          == {k: v["verdict"] for k, v in au["hypotheses"].items()}, (summ["missing"], summ["notes"]))
    never = [n for n in RM.FUNNEL_NEVER if any(k.startswith(n) for k in arts)]
    check("  and no row-level, key or evaluation-descriptor file", not never, never)

    print("F9 recovery and its guards")
    gates = rec_doc["gates"]
    th = pre.raw["recovery_gates"]
    check("recovery is complete, with rows from the veto (R-V) and the panel relabel (R-J)",
          rec_doc["status"] == "complete" and {"R-V", "R-J"} <= got_policies, (rec_doc["status"], got_policies))
    bad_gate = [(r["key"], s) for r in rows for s in r["strata"] if not (gates.get(s) or {}).get("passed")]
    check("every recovered row names only strata whose gate passed", not bad_gate, bad_gate[:5])
    weak = [s for s, g in gates.items() if g.get("passed") and g.get("box_gate") and not (
        g["precision"]["lb"] >= th["box"]["precision_lb_min"] and g["precision"]["estimate"]
        >= th["box"]["precision_point_min"] and (g["precision"]["rogan_gladen"] or {}).get("applied"))]
    check("every passed box gate is Rogan-Gladen corrected with lb >= %.2f and point >= %.2f"
          % (th["box"]["precision_lb_min"], th["box"]["precision_point_min"]), not weak, weak)
    quarantined = set(lk["h6a"]["quarantine"])
    not_rec = set(dom.raw["sources"]["not_recoverable"])
    keys = {r["key"] for r in rows}
    guarded_keys = {"%s__%s" % (sl, st) for sl, st in planted["guarded"]}
    passed_units = set()
    for s, g in gates.items():
        if g.get("passed"):
            passed_units |= {u["unit_id"] for u in frame_rows.get(s, [])}
    guarded_in_passed = {u.split("#")[0][2:] for u in passed_units} & guarded_keys
    licences = dom.raw["sources"].get("licences") or {}
    check("the planted guarded images sit in strata whose gate passed, and the quarantined source has a licence "
          "(so the quarantine alone keeps its images out)",
          guarded_in_passed == guarded_keys and quarantined and quarantined <= set(licences),
          (sorted(guarded_keys - guarded_in_passed)[:5], sorted(quarantined)))
    check("no row of the quarantined source (not even its clean images) or of a not-recoverable source",
          not [r for r in rows if r["source"] in quarantined | not_rec] and not keys & guarded_keys,
          sorted(keys & guarded_keys)[:5])
    check("recovery records the quarantine it applied", rec_doc["quarantined_sources"] == sorted(quarantined))
    dd = _jsonl(r1 / "domain_dev.jsonl")
    dd_keys = {r["key"] for r in dd}
    check("the H10d hold-out exists for every recovered source and no recovered row is in it",
          dd_keys and not keys & dd_keys and {r["source"] for r in rows} <= {r["source"] for r in dd},
          sorted(keys & dd_keys)[:5])
    base_keys = {r["key"] for r in _jsonl(w.step1 / "base_B.jsonl")}
    check("every recovered row is a pool image (no guard-stage image) outside base B",
          keys <= set(truth.pool) and not keys & base_keys)
    guard = C.NeverTrainGuard.load()
    hits, unhashable = guard.check([r["unmasked_image"] for r in rows] + [r["image"] for r in rows])
    check("never-train: every unmasked original and every masked copy is clear of the index (recomputed)",
          not hits and not unhashable, (hits[:3], unhashable[:3]))
    gd = rec_doc["guards"]
    n_masked = sum(1 for r in rows if r.get("masked_boxes"))
    check("the recorded guard covers every row's unmasked image and every masked copy, with no hit (%d of %d "
          "rows masked)" % (n_masked, len(rows)),
          gd["never_train"]["unmasked_checked"] == len(rows) == gd["h6"]["unmasked_checked"]
          and gd["never_train"]["masked_checked"] == n_masked == gd["h6"]["masked_checked"]
          and 0 < n_masked < len(rows)
          and gd["never_train"]["hits"] == 0 and gd["h6"]["copies"] == 0, gd)
    tn = set(dom.target_names)
    knn_targets_only = [j for j in rec_doc.get("mask_judges") or []
                        if j == "J-knn1"]
    check("J-knn1 (a bank of target crops, no non-target answer) does not mask R-J boxes",
          not knn_targets_only, rec_doc.get("mask_judges"))
    wrong_lab, flat_bad, count_bad, relab = [], [], [], collections.Counter()
    for r in rows:
        lab = [ln.split() for ln in open(r["label"]).read().splitlines() if ln.strip()]
        m = truth.meta[r["key"]]
        if len(lab) + len(r["masked_boxes"]) != len(m["boxes"]):
            count_bad.append(r["key"])
        for ln in lab:
            cls, cx, cy = int(ln[0]), float(ln[1]), float(ln[2])
            painted = truth.at(r["unmasked_image"], cx, cy)
            if painted != cls:
                wrong_lab.append((r["key"], cls, painted))
            relab[(r["policy"], dom.class_name(cls))] += 1
        if r["masked_boxes"]:
            a = np.asarray(Image.open(r["image"]).convert("RGB"), dtype=np.float64)
            H, W = a.shape[:2]
            for (_c, cx, cy, bw, bh) in r["masked_boxes"]:
                x0, x1 = int(np.ceil((cx - bw / 2) * W)) + 1, int(np.floor((cx + bw / 2) * W)) - 1
                y0, y1 = int(np.ceil((cy - bh / 2) * H)) + 1, int(np.floor((cy + bh / 2) * H)) - 1
                if a[y0:y1, x0:x1].reshape(-1, 3).std(0).max() > 0.5:
                    flat_bad.append(r["key"])
    check("every recovered label is the painted truth of its box (recovered precision 1.0; %s)"
          % dict(sorted(relab.items())), not wrong_lab, wrong_lab[:5])
    check("labels and masks account for every box of the image", not count_bad, count_bad[:5])
    check("every masked box is one mean-colour fill in the masked copy", not flat_bad, flat_bad[:5])
    kochia = w.pipeline_planted["veto"][0]
    vrow = [r for r in rows if r["policy"] == "R-V"]
    check("R-V keeps the verified target and the named other box, and masks the blocking conflict box",
          vrow and all(len(r["masked_boxes"]) == 1 and truth.at(r["unmasked_image"], r["masked_boxes"][0][1],
                                                                r["masked_boxes"][0][2]) == 8 for r in vrow)
          and bool(kochia))
    changed = [r["key"] for r in rows if sha(truth.pool[r["key"]]["label"]) != truth.pool[r["key"]]["label_sha256"]]
    check("the step-1 labels of every recovered image are unchanged (re-hashed here)",
          not changed and rec_doc["source_labels_unchanged"]["changed"] == 0, changed[:5])

    print("F10 build: realloop_v2 consumes the overlay")
    exp = json.loads((w.inc_dir / "realloop_v2" / "exp.json").read_text())
    inc = exp["step1"]["increment_sources"]
    check("exp.json records the overlay it was built from",
          inc["mode"] == "recovered" and inc["overlay"]["recovery_sha256"] == sha(r1 / "recovery.json")
          and inc["overlay"]["recovered_pool_sha256"] == sha(r1 / "recovered_pool.jsonl"), inc.get("overlay"))
    by_key = {r["key"]: r for r in rows}
    seen, bad_steps = set(), []
    for st in exp["steps"]:
        mrows = C.read_manifest(st["manifest"])
        ks = [r["key"] for r in mrows]
        if (st["clean"] is not False or st.get("kind") != "recovered" or len(ks) != st["n_images"]
                or set(ks) & seen or set(ks) & base_keys or not set(ks) <= set(by_key)
                or any(r["image"] != by_key[r["key"]]["image"] or r["label"] != by_key[r["key"]]["label"]
                       for r in mrows)):
            bad_steps.append(st["name"])
        seen |= set(ks)
    check("6 unclean recovered steps, disjoint from B and from each other, drawn from the overlay's rows",
          len(exp["steps"]) == 6 and not bad_steps, bad_steps)
    arms = docs["step1_r1/arms/arms.json"]
    probs = {}
    for arm, rec_a in sorted(arms["arms"].items()):
        mrows = C.read_manifest(rec_a["path"])
        prov = PL.recovery_provenance(rec_a["path"], mrows, testing=True)
        probs[arm] = (prov or {}).get("problems", ["not a recovery manifest"])
    check("build-baseline's recovery provenance accepts every arm manifest",
          sorted(arms["arms"]) == ["CLASS_ctl", "JUDGE_ctl", "U", "U_ctl"] and not any(probs.values()), probs)


def reruns(run, w, fd, r1, C):
    """Runner §1.2: the same inputs over an existing output are a no-op."""
    print("reruns on the same inputs are no-ops (runner §1.2)")
    before = {p: C.sha256_file(p) for p in (fd / "sample_v1.csv", fd / "frames_v1.json", fd / "audit_v1.json",
                                            fd / "audit_v1.md", fd / "prereg_v1.json", r1 / "recovery.json",
                                            r1 / "recovered_pool.jsonl", fd / "census_v1.json")}
    rcs = [run("census"), run("draw"), run("estimate"),
           run("recover", "--audit", str(fd / "audit_v1.json"), "--maps", str(fd / "class_maps.json"),
               "--policy", "R-A,R-C,R-T,R-V,R-J", out=r1)]
    changed = [p.name for p, h in before.items() if C.sha256_file(p) != h]
    check("census, draw, estimate and recover rerun: exit 0 and every output byte for byte as it was",
          rcs == [0, 0, 0, 0] and not changed, (rcs, changed))


def regressions(fd, dom):
    """Unit checks of the cross-module defects this test found, each one on
    the exact input that broke (so reverting a fix fails here, not only in
    the end-to-end run above)."""
    from weed_optimizer_framework.tools.funnel import FunnelError, SheetError
    from weed_optimizer_framework.tools.funnel import estimate as ES
    from weed_optimizer_framework.tools.funnel import sheets as SH
    from weed_optimizer_framework.tools.inc_autopilot import outcome as OC
    print("regressions of the integration defects")
    ok_names = SH._pred_class_id("MorningGlory", dom, "t") == dom.class_id("MorningGlory")
    ok_ids = SH._pred_class_id("1", dom, "t") == 1
    try:
        SH._pred_class_id("Tomato", dom, "t")
        refused = False
    except SheetError:
        refused = True
    check("sheets reads the frames' pred as strata writes it (a class name), or an id; refuses anything else",
          ok_names and ok_ids and refused)
    e = ES.ht([1.0] * 68, [0.8] * 68, 81)
    check("an HT stratum whose every labelled unit is positive has an informative interval (Korn-Graubard n)",
          e["n_eff"] == 68 and e["interval"][0] > 0.9, e)
    c = ES.combine([{"N": 40, "theta": 1.0, "var": 0.002, "n": 30}])
    check("  and so has an aggregate at 1.0 with a positive variance", c["interval"][0] > 0.85, c)
    g = {"answer": "Purslane", "answer_level": "species", "answer_taxon": "Portulaca oleracea", "box_ok": "yes"}
    ctx_event = ES.Context.event.__get__(_EventCtx(dom))
    try:
        got = (ctx_event("label:Purslane", g, None, "species"), ctx_event("label:Waterhemp", g, None, "species"))
    except FunnelError as e:
        got = "%s: %s" % (type(e).__name__, e)
    check("estimate's event 'label:<class>' (a card-mapped unit's box gate) is answered as that class, box valid",
          got == (1, 0), got)
    sid = "G1/frame=noinfo/status=numeric/pred=Purslane"
    audit = {"strata": [{"stratum": sid, "event": "pred", "interval": [0.9, 1.0]},
                        {"stratum": sid, "event": "target", "interval": [0.0, 0.1]}]}
    v = OC._prediction_outcome({"metric": "purity", "stratum": sid, "direction": "above", "threshold": 0.5},
                               audit, None)
    check("the DA's purity prediction is scored on the stratum's target share, not on a gate's estimate listed "
          "before it", v[0] == "wrong", v)


class _EventCtx(object):
    """The part of estimate.Context that event() reads."""

    def __init__(self, dom):
        from weed_optimizer_framework.tools.funnel import estimate as ES
        self.domain = dom
        self._targets = set(dom.target_names)
        self._target_genera = {ES._genus(t.get("taxon")) for t in dom.targets if ES._genus(t.get("taxon"))}
        self._unsure_genera = set(dom.genus_unsure())

    def target_answer(self, g, level):
        from weed_optimizer_framework.tools.funnel import estimate as ES
        return ES.Context.target_answer(self, g, level)


def report(w, fd, r1):
    au = json.loads((fd / "audit_v1.json").read_text())
    print("audit valid %s stop %s" % (au["valid"], au["stop"]))
    for h, v in sorted(au["hypotheses"].items()):
        print("  %-5s %-13s %s" % (h, v["verdict"], (v.get("why") or "")[:160]))
    if (r1 / "recovery.json").exists():
        rec = json.loads((r1 / "recovery.json").read_text())
        print("recovery", rec.get("status"), rec.get("counts"))
        for s, g in sorted((rec.get("gates") or {}).items()):
            print("  gate %s %s %s" % (s, g.get("passed"), g.get("why", "")[:150]))


if __name__ == "__main__":
    main()
