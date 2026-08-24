"""Per-source label sample-audit (SUPERWEED_PLAN S1's last missing mechanism).

Samples images from a registry source, cross-checks its labels against an
independent detector, flags geometry outliers, and saves a montage a human can
eyeball — then writes the verdict into the source's registry scorecard so pool
admission is a recorded measurement instead of a supervisor's memory.

**Why OWLv2 is used the way it is.** This project's own benchmark measured OWLv2
at recall 0.943 / precision 0.194 on cwd12. It is therefore useless as ground
truth (four in five of its boxes are false) but excellent as a *recall* probe: if
an open-vocabulary detector that finds 94 % of real weeds sees nothing at all where
a label claims an object, that label is suspect. So the audit reports

    suspect_rate = GT boxes with no OWLv2 box at IoU >= 0.3 / all sampled GT boxes

and never treats an OWLv2 box as evidence that a label is *correct*. A source
passes when `1 - suspect_rate >= 0.90` (the plan's ≥90 % precision bar), and the
verdict is always stored with the sample size so it can be judged.

Geometry checks are independent of any model: degenerate boxes (zero/negative
area), out-of-range coordinates, and extreme aspect/area outliers.

Usage (repo root, GPU node for the OWLv2 pass):
    python -m weed_optimizer_framework.tools.sample_audit audit <slug> [--n 25]
    python -m weed_optimizer_framework.tools.sample_audit audit-all [--limit 10]
    python -m weed_optimizer_framework.tools.sample_audit report
"""
import argparse
import json
import os
import random
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REG = os.path.join(REPO, "results", "framework", "dataset_registry.json")
OUT_DIR = os.path.join(REPO, "results", "framework", "sample_audits")

PASS_BAR = 0.90          # plan's ≥90 % audited precision
IOU_MATCH = 0.30         # generous: we are asking "did it see ANYTHING here"
OWL_CONF = 0.10          # low bar on purpose — this is a recall probe
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _registry():
    return json.load(open(REG))


def _label_pairs(root, cap=4000):
    """(image, label) pairs under a dataset root, bounded, images-dir-safe."""
    labels = {}
    for dirpath, dirnames, filenames in os.walk(root):
        base = os.path.basename(dirpath).lower()
        if base.startswith("label"):
            for f in filenames:
                if f.endswith(".txt"):
                    labels.setdefault(os.path.splitext(f)[0], os.path.join(dirpath, f))
                    if len(labels) >= cap:
                        break
        if len(labels) >= cap:
            break
    pairs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath).lower().startswith("label"):
            continue
        for f in filenames:
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXT and stem in labels:
                pairs.append((os.path.join(dirpath, f), labels[stem]))
                if len(pairs) >= cap:
                    return pairs
    return pairs


def _read_yolo(path):
    out = []
    try:
        for line in open(path):
            p = line.split()
            if len(p) >= 5:
                try:
                    out.append((int(float(p[0])), *[float(x) for x in p[1:5]]))
                except ValueError:
                    continue
    except Exception:
        pass
    return out


def _geometry_flags(boxes):
    bad = 0
    for _c, x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            bad += 1
        elif not (0 <= x <= 1 and 0 <= y <= 1):
            bad += 1
        elif w > 1.0 or h > 1.0:
            bad += 1
        elif w * h < 1e-6:                      # sub-pixel at any sane resolution
            bad += 1
        elif max(w / max(h, 1e-9), h / max(w, 1e-9)) > 50:
            bad += 1
    return bad


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def audit(slug, n=25, montage=True):
    reg = _registry()
    info = (reg.get("datasets") or {}).get(slug)
    if not isinstance(info, dict):
        raise SystemExit("unknown slug: " + slug)
    root = info.get("local_path")
    pairs = _label_pairs(root) if root else []
    if not pairs:
        return _store(slug, {"ok": False, "reason": "no image/label pairs found",
                             "root": root})
    random.seed(1234)                            # reproducible sample
    sample = random.sample(pairs, min(n, len(pairs)))

    from PIL import Image
    import numpy as np, torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection

    names = [str(c) for c in (info.get("class_names") or [])]
    prompt = [["a photo of a weed", "a photo of a crop plant"]]
    if names:
        prompt = [["a photo of a " + nm for nm in names[:12]]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-large-patch14-ensemble").to(dev).eval()

    tot_gt = suspect = geom_bad = empty_lbl = 0
    per_image, thumbs = [], []
    for img_path, lbl_path in sample:
        boxes = _read_yolo(lbl_path)
        if not boxes:
            empty_lbl += 1
            continue
        try:
            im = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        W, H = im.size
        gt = [((x - w / 2) * W, (y - h / 2) * H, (x + w / 2) * W, (y + h / 2) * H)
              for _c, x, y, w, h in boxes]
        geom_bad += _geometry_flags(boxes)
        with torch.no_grad():
            inp = proc(text=prompt, images=im, return_tensors="pt").to(dev)
            out = model(**inp)
            res = proc.post_process_grounded_object_detection(
                out, target_sizes=torch.tensor([[H, W]]).to(dev), threshold=OWL_CONF)[0]
        owl = [tuple(float(v) for v in b) for b in res["boxes"].cpu().numpy()]
        miss = sum(1 for g in gt if max([_iou(g, o) for o in owl] or [0.0]) < IOU_MATCH)
        tot_gt += len(gt)
        suspect += miss
        per_image.append({"image": os.path.basename(img_path), "gt": len(gt),
                          "owl": len(owl), "unseen_gt": miss})
        if montage and len(thumbs) < 12:
            th = im.copy()
            th.thumbnail((320, 320))
            thumbs.append(th)

    rate = (suspect / tot_gt) if tot_gt else 1.0
    verdict = {
        "ok": True,
        "sampled_images": len(sample),
        "images_with_labels": len(sample) - empty_lbl,
        "gt_boxes": tot_gt,
        "unseen_by_owlv2": suspect,
        "suspect_rate": round(rate, 4),
        "audited_precision": round(1 - rate, 4),
        "geometry_bad_boxes": geom_bad,
        "empty_label_files": empty_lbl,
        "passes_bar": bool(tot_gt and (1 - rate) >= PASS_BAR and geom_bad == 0),
        "bar": PASS_BAR,
        "method": ("OWLv2 recall probe (conf %.2f, IoU %.2f). OWLv2 is 0.943-recall / "
                   "0.194-precision on cwd12, so it is used only to flag labels it "
                   "cannot see, never to confirm labels it agrees with."
                   % (OWL_CONF, IOU_MATCH)),
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "per_image": per_image[:25],
    }
    if montage and thumbs:
        try:
            os.makedirs(OUT_DIR, exist_ok=True)
            cols = 4
            rows = (len(thumbs) + cols - 1) // cols
            cw = max(t.width for t in thumbs)
            ch = max(t.height for t in thumbs)
            sheet = Image.new("RGB", (cols * cw, rows * ch), (18, 22, 30))
            for i, t in enumerate(thumbs):
                sheet.paste(t, ((i % cols) * cw, (i // cols) * ch))
            p = os.path.join(OUT_DIR, slug + ".jpg")
            sheet.save(p, quality=82)
            verdict["montage"] = p
        except Exception as e:
            verdict["montage_error"] = str(e)[:120]
    return _store(slug, verdict)


def _store(slug, verdict):
    from .registry_lock import update_registry

    def _apply(r):
        info = r.get("datasets", {}).get(slug)
        if isinstance(info, dict):
            info.setdefault("scorecard", {})["sample_audit"] = verdict
        return r
    update_registry(REG, _apply)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, slug + ".json"), "w") as f:
        json.dump(verdict, f, indent=1)
    print("%-46s precision=%s pass=%s (gt=%s unseen=%s geom_bad=%s)"
          % (slug[:46], verdict.get("audited_precision"), verdict.get("passes_bar"),
             verdict.get("gt_boxes"), verdict.get("unseen_by_owlv2"),
             verdict.get("geometry_bad_boxes")))
    return verdict


def audit_all(limit=10, n=25, only_unaudited=True):
    reg = _registry()
    todo = []
    for slug, info in (reg.get("datasets") or {}).items():
        if not isinstance(info, dict) or info.get("status") == "quarantined":
            continue
        card = info.get("scorecard") or {}
        if only_unaudited and (card.get("sample_audit") or {}).get("ok"):
            continue
        if not card.get("in_audited_pool"):
            continue
        todo.append(slug)
    todo = todo[:limit]
    print("auditing %d source(s)" % len(todo))
    for slug in todo:
        try:
            audit(slug, n=n)
        except Exception as e:
            print("  %-46s FAILED %s: %s" % (slug[:46], type(e).__name__, str(e)[:90]))


def report():
    reg = _registry()
    rows = []
    for slug, info in (reg.get("datasets") or {}).items():
        if not isinstance(info, dict):
            continue
        a = ((info.get("scorecard") or {}).get("sample_audit") or {})
        if a.get("ok"):
            rows.append((slug, a))
    if not rows:
        print("no sample audits recorded yet")
        return
    rows.sort(key=lambda r: r[1].get("audited_precision", 0))
    print("SAMPLE AUDITS (%d source(s), bar %.2f)" % (len(rows), PASS_BAR))
    for slug, a in rows:
        print("  %-46s precision=%.3f %s  gt=%-5s geom_bad=%s"
              % (slug[:46], a.get("audited_precision", 0),
                 "PASS" if a.get("passes_bar") else "FAIL",
                 a.get("gt_boxes"), a.get("geometry_bad_boxes")))
    failed = [s for s, a in rows if not a.get("passes_bar")]
    if failed:
        print("\nbelow the bar — re-label or quarantine: %s" % failed)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("audit"); a1.add_argument("slug"); a1.add_argument("--n", type=int, default=25)
    a2 = sub.add_parser("audit-all"); a2.add_argument("--limit", type=int, default=10); a2.add_argument("--n", type=int, default=25)
    sub.add_parser("report")
    a = ap.parse_args()
    if a.cmd == "audit":
        audit(a.slug, n=a.n)
    elif a.cmd == "audit-all":
        audit_all(limit=a.limit, n=a.n)
    else:
        report()
