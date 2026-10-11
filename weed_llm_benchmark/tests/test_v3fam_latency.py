#!/usr/bin/env python3
"""V3F D7: the standalone latency script for the lab RTX 3060
(tools/latency/two_stage_latency.py; docs/CONTINUOUS_LOOP.md, "Amendment
(2026-10-11): V3F", V3F.5 D7). No GPU and no model: the script is loaded from
its file, as the lab runs it.

- the protocol's constants (550 dev images: 50 warm-up, 500 timed; batch 1,
  FP16; conf 0.25 and tau 0.1 / 0.5; NMS IoU 0.7, max_det 300; the 20-image
  software check at conf 0.001 within 1 px, batch 32; Ultralytics 8.4.37,
  open_clip 3.3.0, BioCLIP-2 at snapshot 2957b322; the pinned weights; D2's
  backbones) and that it imports nothing of the platform;
- its copies of E3's helpers equal E3's own: stable_int, the square crop
  (pixel for pixel), degenerate, normalise, the classifier's probabilities,
  the L2 norm, one box per location, the inverse letterbox;
- the image ranking (deterministic, the first 50 and the next 500), the dev
  manifest read without its label paths (no label opened), the images' sha256
  check, the pinned weights, the BioCLIP-2 snapshot, the GPU wait (waits,
  never stops a process, gives up writing nothing), the summaries, the
  software check's comparison, the table.
(The software check's validator path against the family stage1 pass runs in
tests/test_inc2_v3fam_systems.py, in the synthetic world.)

Run:  python3 tests/test_v3fam_latency.py
"""
import hashlib
import importlib.util
import json
import pathlib
import re
import shutil
import sys
import time
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import test_inc2_train as W  # noqa: E402  (sets INC_DIR / REPO first)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from weed_optimizer_framework.tools import semisup_labeler as SL  # noqa: E402
from weed_optimizer_framework.tools.inc import common as C  # noqa: E402
from weed_optimizer_framework.tools.inc import scorer as S  # noqa: E402
from weed_optimizer_framework.tools.inc import verify as V  # noqa: E402
from weed_optimizer_framework.tools.inc2 import twostage as TS  # noqa: E402

FAILURES = W.FAILURES
check = W.check
SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "weed_optimizer_framework" / "tools" / "latency" / \
    "two_stage_latency.py"


def load():
    spec = importlib.util.spec_from_file_location("v3fam_two_stage_latency", str(SCRIPT))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TL = load()


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except TL.LatencyRefused as e:
        return e
    return None


def t_constants():
    print("the protocol (V3F.5 D7)")
    check("550 dev images ranked by stable_int('inc2/v3fam/d7/<key>'): 50 warm-up, 500 timed; 640 px, conf 0.25 (tau "
          "0.1 and 0.5 beside), NMS IoU 0.7, max_det 300; the software check on 20 images at conf 0.001 within 1 px, "
          "batch 32; Ultralytics 8.4.37 and open_clip 3.3.0; BioCLIP-2 at snapshot 2957b322",
          TL.SEED_TEXT == "inc2/v3fam/d7/%s" and TL.WARMUP == 50 and TL.TIMED == 500 and TL.IMGSZ == 640
          and TL.CONF == 0.25 and TL.TAUS == (0.1, 0.5) and TL.NMS_IOU == 0.7 and TL.MAX_DET == 300
          and TL.CHECK_IMAGES == 20 and TL.CHECK_CONF == 0.001 and TL.CHECK_TOL_PX == 1.0 and TL.CHECK_BATCH == S.BATCH
          and TL.ULTRALYTICS == S.PINNED_ULTRALYTICS == "8.4.37" and TL.OPEN_CLIP == "3.3.0"
          and TL.BIOCLIP == V.EMBEDDER_NAME and TL.BIOCLIP_SNAPSHOT.startswith("2957b322")
          and TL.DIFFERENT == "different software")
    check("the pinned files: E1-B s0 (d84aa235...fc60), s1, s2, b_v2_m640 s0 (1d9027f6...), E3's classifier.npz "
          "(4905e4a6...f42b)", TL.PINNED["stage1"].startswith("d84aa235") and TL.PINNED["stage1"].endswith("fc60")
          and TL.PINNED["reference"].startswith("1d9027f6") and TL.PINNED["classifier"].startswith("4905e4a6")
          and TL.PINNED["classifier"].endswith("f42b") and len(set(TL.PINNED.values())) == 5
          and all(re.fullmatch(r"[0-9a-f]{64}", v) for v in TL.PINNED.values()))
    ids = [x[2] for x in TL.D2_BACKBONES]
    check("D2's seven other frozen backbones are timed per crop (FT-ResNet-50 as ResNet-50, BioCLIP-2@640 as "
          "BioCLIP-2)", ids == ["hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K", "openai/clip-vit-large-patch14",
                                "facebook/dinov2-large", "facebook/dinov2-base", "google/siglip2-so400m-patch14-224",
                                "hf-hub:timm/resnet50.a1_in1k", "hf-hub:imageomics/bioclip"])
    src = SCRIPT.read_text()
    imports = re.findall(r"^\s*(?:from|import)\s+([A-Za-z_][\w.]*)", src, flags=re.M)
    check("standalone: it imports nothing of the platform (%d import lines)" % len(imports),
          not any(i.startswith(("weed_optimizer_framework", ".")) for i in imports) and "from ." not in src)


def t_helpers():
    print("E3's helpers, copied, equal E3's own")
    keys = ["k%04d" % i for i in range(200)] + ["20210903_iPhone11Pro_BT_148", "ood22__x_1"]
    check("stable_int", all(TL.stable_int(TL.SEED_TEXT % k) == C.stable_int(TL.SEED_TEXT % k) for k in keys))
    rng = np.random.default_rng(3)
    im = Image.fromarray((rng.random((97, 131, 3)) * 255).astype(np.uint8))
    W_, H_ = im.size
    boxes = [(0.5, 0.5, 0.2, 0.3), (0.02, 0.03, 0.05, 0.06), (0.98, 0.97, 0.1, 0.2), (0.5, 0.5, 1.2, 0.9),
             (0.3, 0.7, 0.001, 0.002)]
    same = [np.array_equal(np.asarray(TL.cut(im, *b)), np.asarray(SL._cut(im, {"cx": b[0], "cy": b[1], "w": b[2],
                                                                                "h": b[3], "W": W_, "H": H_})))
            for b in boxes]
    check("the square crop (margin 10 %%, grey padding, 224 px bicubic) pixel for pixel, %d boxes incl. the edges"
          % len(boxes), all(same) and TL.CROP_PX == SL.CROP_PX and TL.MARGIN == SL.MARGIN, same)
    check("square_box, degenerate and normalise", all(TL.square_box(*b, W_, H_) == SL._square_box(*b, W_, H_)
                                                       for b in boxes)
          and all(TL.degenerate(*b, W_, H_) == TS.degenerate(*b, W_, H_) for b in boxes)
          and np.array_equal(TL.normalise([[1, 2, 30, 40], [0, 0, 131, 97]], W_, H_),
                             TS.normalise([[1, 2, 30, 40], [0, 0, 131, 97]], W_, H_)))
    X = rng.normal(size=(7, 16))
    Wm, b = rng.normal(size=(12, 16)), rng.normal(size=12)
    cls = np.array([c for c in range(13) if c != 4])
    check("the classifier's probabilities and the L2 norm", np.array_equal(TL.proba(X, Wm, b, cls),
                                                                          TS.proba(X, Wm, b, cls))
          and np.array_equal(TL.l2(X), V._norm(X)))
    xy = np.round(rng.random((40, 4)) * 50, 1)
    xy[10:20] = xy[:10]
    q = np.round(rng.random(40), 2)
    rp, ori = ((0.5, 0.25), (3.0, 7.0)), (480, 640)
    check("one box per location and the inverse letterbox", np.array_equal(TL.one_per_box(xy, q), S.one_per_box(xy, q))
          and np.array_equal(TL.to_original(xy * 10, rp, ori), TS.to_original(xy * 10, rp, ori)))


def t_inputs():
    print("inputs: dev pixels only")
    d = W.TMP / "lat"
    rows = []
    for i in range(560):
        p = d / "img" / ("dv_%04d.jpg" % i)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"jpeg%d" % i)
        rows.append({"key": "dv_%04d" % i, "image": "/cluster/dev/dv_%04d.jpg" % i,
                     "label": str(d / "lab" / ("dv_%04d.txt" % i)), "sha256": hashlib.sha256(b"jpeg%d" % i).hexdigest(),
                     "session": "s", "source": "cwd12"})
    man = d / "dev.jsonl"
    man.write_text("".join(json.dumps(r) + "\n" for r in rows))
    opened = []

    def hook(event, args):
        if event == "open" and args and isinstance(args[0], str) and "/lab/" in args[0]:
            opened.append(args[0])
    sys.addaudithook(hook)
    got = TL.dev_rows(str(man), [("/cluster/dev/", str(d / "img") + "/")])
    check("the manifest read keeps key, image (path-mapped) and sha256 only; no label path kept or opened",
          len(got) == 560 and set(got[0]) == {"key", "image", "sha256"} and got[0]["image"].startswith(str(d))
          and not opened, opened[:3])
    warm, timed = TL.ranked(got)
    order = sorted(got, key=lambda r: (C.stable_int("inc2/v3fam/d7/%s" % r["key"]), r["key"]))
    check("the ranking: the first 50 by stable_int warm up, the next 500 are timed; deterministic",
          [r["key"] for r in warm] == [r["key"] for r in order[:50]]
          and [r["key"] for r in timed] == [r["key"] for r in order[50:550]] and TL.ranked(got[::-1])[1] == timed)
    check("  fewer than 550 images refused", refused(TL.ranked, got[:549]) is not None)
    check("the images' bytes against the manifest's sha256 (560 checked)", TL.check_images(got) == 560)
    (d / "img" / "dv_0007.jpg").write_bytes(b"another image")
    e = refused(TL.check_images, got)
    check("  another file is refused", e is not None and "dv_0007" in str(e), e)
    w = d / "w.pt"
    w.write_bytes(b"not the weights")
    e = refused(TL.check_pinned, {"stage1": str(w)})
    check("the pinned files: a file hashing otherwise is refused", e is not None and "stage1" in str(e), e)
    cache = d / "hf"
    ref = cache / "models--imageomics--bioclip-2" / "refs" / "main"
    ref.parent.mkdir(parents=True)
    ref.write_text("0" * 40)
    e = refused(TL.bioclip_snapshot, cache)
    ref.write_text(TL.BIOCLIP_SNAPSHOT + "\n")
    check("BioCLIP-2's cache: another snapshot refused, the pinned one read", e is not None
          and TL.bioclip_snapshot(cache) == TL.BIOCLIP_SNAPSHOT, e)


class FakeSmi(object):
    def __init__(self, busy_polls):
        self.busy, self.calls = busy_polls, []

    def __call__(self, cmd, capture_output=True, text=True):
        self.calls.append(cmd)
        out = ""
        if "--query-compute-apps=pid,process_name,used_memory" in cmd and self.busy > 0:
            self.busy -= 1
            out = "4242, /usr/bin/ollama, 5000\n"
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")


def t_gpu():
    print("the GPU: wait for it, never stop anything")
    smi = FakeSmi(2)
    slept = []
    waited = TL.wait_gpu_free(600, poll_s=60, run=smi, sleep=slept.append, own_pid=1)
    check("busy twice (another process), then free: waited 2 x 60 s, only nvidia-smi queried, no process signalled",
          waited == 120 and slept == [60, 60] and all(c[0] == "nvidia-smi" for c in smi.calls))
    smi2 = FakeSmi(10 ** 6)
    e = refused(TL.wait_gpu_free, 180, poll_s=60, run=smi2, sleep=lambda s: None, own_pid=1)
    check("  busy past the limit: refused (nothing timed, nothing written)", e is not None and "busy" in str(e), e)
    check("  this process's own entry does not count", TL.wait_gpu_free(10, run=FakeSmi(5), sleep=lambda s: None,
                                                                         own_pid=4242) == 0)


def t_reporting():
    print("summaries, the software check's comparison, the table")
    a = np.random.default_rng(1).random(500) * 30
    s = TL.summary(a)
    check("median, p90, p99 as numpy's linear percentiles", s["n"] == 500 and s["median"] == float(np.percentile(a, 50))
          and s["p90"] == float(np.percentile(a, 90)) and s["p99"] == float(np.percentile(a, 99))
          and TL.summary([]) == {"n": 0})
    stored = {"keys": np.array(["a", "b", "c"]), "img": np.array([0, 0, 1]),
              "xyxy": np.array([[1, 1, 5, 5], [2, 2, 8, 9], [3, 3, 4, 4.0]]), "q": np.array([0.9, 0.5, 0.3])}
    got = {"a": (np.array([[2, 2, 8.4, 9], [1, 1, 5, 5.2]]), np.array([0.5, 0.9])), "b": (np.array([[3, 3, 4, 4.0]]),
                                                                                             np.array([0.3]))}
    ok = TL.compare_stage1(got, stored, ["a", "b"])
    bad = TL.compare_stage1(dict(got, a=(got["a"][0] + 1.5, got["a"][1])), stored, ["a", "b"])
    fewer = TL.compare_stage1(dict(got, a=(got["a"][0][:1], got["a"][1][:1])), stored, ["a"])
    miss = TL.compare_stage1(got, stored, ["a", "c"])
    check("the software check: boxes within 1 px in any row order match; 1.5 px off, another count or a missing image "
          "do not", ok["matched"] and abs(ok["max_diff_px"] - 0.4) < 1e-9 and not bad["matched"]
          and not fewer["matched"] and not miss["matched"], (ok, bad["max_diff_px"]))
    doc = {"created_utc": "2026-10-11T00:00:00Z", "software_label": "pinned",
           "timings": {"two-stage q>=0.25": {"total": s, "boxes_per_image": s}, "D3": "not run"},
           "backbones": {"DINOv2-B": {"per_crop": s, "per_image": s}, "SigLIP2": {"status": "not available"}}}
    t = TL.table(doc)
    check("the table: every stage's median / p90 / p99, each backbone's per-crop cost or 'not available'",
          "| two-stage q>=0.25 | total |" in t and "| DINOv2-B |" in t and "not available" in t)


TESTS = ("t_constants", "t_helpers", "t_inputs", "t_gpu", "t_reporting")


def main():
    t0 = time.time()
    try:
        for name in TESTS:
            globals()[name]()
    finally:
        shutil.rmtree(W.TMP, ignore_errors=True)
    print("\n%d failure(s) in %.0fs" % (len(FAILURES), time.time() - t0))
    for x in FAILURES:
        print("  FAILED: %s" % x)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
