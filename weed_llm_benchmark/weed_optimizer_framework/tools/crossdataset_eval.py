"""Cross-dataset transfer of the deployed cwd12 model — BEST_MODEL_CARD limit #1.

The card's first honest limit says every number is CottonWeedDet12 and that
cross-dataset generalisation is *not* established. This measures it, on the one
harvested source that cleared the calibrated sample audit at 1.0000 precision:
`project_agml__imageweeds_weed_detection` (HF Project-AgML/ImageWeeds_weed_detection,
CC BY 4.0, 3,208 images / 6,932 boxes, ground-level field photography).

Design decisions a reviewer would probe, settled up front:

* LEAK CHECK FIRST, content-level. The web pool is full of re-exports (the merge
  funnel measured 44,750 cross-dataset duplicates), and ImageWeeds' filenames are
  anonymised, so stems prove nothing. Every ImageWeeds image is dHashed and
  compared against BOTH the cwd12 train split and the sealed holdout at Hamming
  distance <= HAMMING_LEAK (re-encoded JPEGs shift a few bits; exact-match-only
  would fake a clean bill). Collisions are excluded from the transfer set and
  reported in the artifact either way.

* TWO measurements, not one:
  - class-agnostic weed localisation: every prediction and every GT box mapped to
    one class. Answers "does the detector find weeds at all in foreign imagery?"
    ImageWeeds' 3 `corn` boxes (a crop, id 2) are dropped from GT — the model was
    never asked to find crops.
  - Ragweed-only, species-matched: ImageWeeds id 3 `ragweed`, assumed to be
    common ragweed (Ambrosia artemisiifolia; the dataset card gives only the
    common name), against the model's Ragweed class. The model's class is found
    by species: species_names_for(model.names), so a head that carries the
    legacy labels resolves to the id holding Ragweed (cwd12 id 5 for the sealed
    seeds, whose legacy label there is "Nutsedge"; trainer slot 11 for merged
    slot-space heads), never to the id whose stored name is "Ragweed" (that one
    holds Sicklepod). `redrootpigweed` is Amaranthus retroflexus -- same genus as
    PalmerAmaranth (A. palmeri), NOT the same species -- and is deliberately not
    mapped.
    v3.60.0: before this, the arm scored the legacy "Ragweed" class, i.e. a
    Sicklepod detector against ragweed boxes, and its in-domain reference was
    Sicklepod's holdout AP. Artifacts written before v3.60.0 (keys iw_ragweed /
    holdout_ragweed) measure Sicklepod; the corrected arm writes
    iw_ragweed_species / holdout_ragweed_species.

* IN-DOMAIN REFERENCE UNDER THE SAME METRIC. Both measurements are also run on
  the cwd12 sealed holdout with the same checkpoints, matcher, conf floor and
  image size, so the transfer gap is (same metric, different data) and never
  (different metric). The matcher is wbf_tta_eval's — the TTA ceiling runs
  already measured its offset vs the Ultralytics validator, so these numbers
  compose with that work rather than forking a third protocol.

* ONE ID SPACE PER COMPARISON. The sealed holdout labels are raw cwd12 ids,
  while a checkpoint may be in cwd12-id order (the s3 seeds) or trainer slot
  order (merged heads). Predictions are translated to cwd12 ids through the
  species of each model class before they are compared with holdout GT, so a
  slot-11 prediction is scored against raw id 5 GT, never slot 5 against id 5.
  The detected label space and the ids used are recorded per checkpoint.

* n=3 seeds, mean±std, one inference pass per (seed, image) at conf 0.001; all
  four metrics are post-hoc filters over the same cached raw predictions.

Class-name provenance: the ImageWeeds label files ship with no yaml; ids are from
the HF dataset card's dataset_info (README.md, checked 2026-08-26):
0 horseweed, 1 kochia, 2 corn, 3 ragweed, 4 redrootpigweed.

Artifact: results/framework/s6_crossdataset_imageweeds_species.json (XDS_OUT
overrides it). An artifact written before v3.60.0 at the target path is never
overwritten; the run writes next to it with a `_species` suffix instead.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

from weed_optimizer_framework.tools.cwd12_species import (
    CWD12_SPECIES, TRAINER_SLOT_SPECIES, species_names_for, species_of)
from weed_optimizer_framework.tools.wbf_tta_eval import (
    _resolve_holdout, compute_map, predict_one_scale, read_yolo_labels)

REPO = Path(os.environ.get(
    "REPO_ROOT", "/ocean/projects/cis240145p/byler/harry/weed_llm_benchmark"))
IW_DIR = REPO / "datasets" / "project_agml__imageweeds_weed_detection"
# Which checkpoints to transfer. The default is the three sealed cwd12 seeds, the
# comparison this file was written for. XDS_WEIGHTS overrides it with
# "label=path,label=path,..." so the same protocol -- same leak check, same
# matcher, same conf and imgsz on both sides -- can be pointed at any other set of
# checkpoints without a second implementation drifting away from this one.
_DEFAULT_SEEDS = (101, 102, 103)
_env_w = (os.environ.get("XDS_WEIGHTS") or "").strip()
if _env_w:
    WEIGHTS = {}
    for part in _env_w.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            WEIGHTS[k.strip()] = Path(v.strip())
    SEEDS = tuple(WEIGHTS)
else:
    SEEDS = _DEFAULT_SEEDS
    WEIGHTS = {s: REPO / f"results/framework/s3_yolo11n/s{s}/weights/best.pt"
               for s in SEEDS}
OUT = Path(os.environ.get("XDS_OUT") or
           (REPO / "results/framework/s6_crossdataset_imageweeds_species.json"))
CONF, IMGSZ, IOU_NMS = 0.001, 640, 0.6
HAMMING_LEAK = 6
IW_RAGWEED_ID, IW_CORN_ID = 3, 2
# The species of the per-species arm, and its id in the sealed holdout's label
# files (raw cwd12 ids).
ARM_SPECIES = "Ragweed"
HOLDOUT_ARM_ID = CWD12_SPECIES.index(ARM_SPECIES)


def label_space(names):
    """Which id space a head's class list is in, after legacy translation:
    'cwd12_id' (cwd12 label-id order), 'trainer_slot' (mega_trainer slots 0-11,
    optionally followed by aux classes) or 'other'."""
    seq = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    seq = list(species_names_for(seq))
    if seq[:12] == CWD12_SPECIES:
        return "cwd12_id"
    if seq[:12] == TRAINER_SLOT_SPECIES:
        return "trainer_slot"
    return "other"


def model_to_cwd12(names):
    """{model class id: cwd12 id} for every model class that is a cwd12 species.

    A whole legacy list is translated by species_names_for; any other class name
    is a real name and goes through species_of. Classes that are no cwd12
    species (aux slots, crops, other weeds) are absent."""
    if not isinstance(names, dict):
        names = dict(enumerate(names))
    sp = species_names_for(names)
    out = {}
    for k, v in sp.items():
        s = v if v in CWD12_SPECIES else species_of(v)
        if s is not None:
            out[int(k)] = CWD12_SPECIES.index(s)
    return out


def output_path(out):
    """`out`, unless it holds a pre-v3.60.0 artifact (per-seed key iw_ragweed,
    whose species arm measured Sicklepod): that file is kept and the new result
    goes next to it with a `_species` suffix."""
    try:
        old = json.loads(Path(out).read_text())
    except (OSError, ValueError):
        return Path(out)
    rows = old.get("per_seed") if isinstance(old, dict) else None
    if isinstance(rows, dict) and any(
            isinstance(r, dict) and "iw_ragweed" in r for r in rows.values()):
        out = Path(out)
        return out.with_name(out.stem + "_species" + out.suffix)
    return Path(out)


def dhash64(img):
    """Standard 8x8 difference hash -> uint64. Both sides of every comparison
    are hashed by this same function in this same run, so there is no
    cross-artifact compatibility concern."""
    g = np.asarray(img.convert("L").resize((9, 8)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def hash_dir(paths, label):
    from PIL import Image
    out, bad, t0 = [], 0, time.time()
    for i, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                out.append(dhash64(im))
        except Exception:
            bad += 1
            out.append(np.uint64(0))
        if (i + 1) % 1000 == 0:
            print(f"[hash] {label} {i+1}/{len(paths)} ({time.time()-t0:.0f}s)",
                  flush=True)
    if bad:
        print(f"[hash] {label}: {bad} unreadable")
    return np.array(out, dtype=np.uint64)


def min_hamming(a, b):
    """Per-row minimum Hamming distance from a (N,) to b (M,), uint64."""
    x = a[:, None] ^ b[None, :]                     # (N, M) uint64
    u8 = x.view(np.uint8).reshape(len(a), len(b), 8)
    return np.unpackbits(u8, axis=2).sum(axis=2).min(axis=1)


def main():
    from ultralytics import YOLO

    iw_imgs = sorted((IW_DIR / "images").glob("*.jpg"))
    iw_lbls = [IW_DIR / "labels" / (p.stem + ".txt") for p in iw_imgs]
    assert len(iw_imgs) == 3208, f"expected 3208 ImageWeeds images, found {len(iw_imgs)}"

    hold_dirs, cwd_names = _resolve_holdout(str(REPO / "cwd12_sealed.yaml"))
    hold_imgs = sorted(p for d in hold_dirs for p in d.glob("*.jpg"))
    hold_lbls = [Path(str(p.parent).replace("/images", "/labels")) / (p.stem + ".txt")
                 for p in hold_imgs]
    train_imgs = sorted((REPO / "downloads/cottonweeddet12/train/images").glob("*.jpg"))
    print(f"[sets] imageweeds={len(iw_imgs)}  holdout={len(hold_imgs)}  "
          f"cwd12-train={len(train_imgs)}")

    # ---- leak check ---------------------------------------------------------
    h_iw = hash_dir(iw_imgs, "imageweeds")
    h_tr = hash_dir(train_imgs, "cwd12-train")
    h_ho = hash_dir(hold_imgs, "cwd12-holdout")
    d_tr, d_ho = min_hamming(h_iw, h_tr), min_hamming(h_iw, h_ho)
    d_any = np.minimum(d_tr, d_ho)
    leak_mask = d_any <= HAMMING_LEAK
    leak = {
        "hamming_threshold": HAMMING_LEAK,
        "vs_train": {f"<={t}": int((d_tr <= t).sum()) for t in (0, 4, 6, 10)},
        "vs_holdout": {f"<={t}": int((d_ho <= t).sum()) for t in (0, 4, 6, 10)},
        "excluded_images": int(leak_mask.sum()),
        "excluded_names": [iw_imgs[i].name for i in np.where(leak_mask)[0][:50]],
    }
    print(f"[leak] {leak}")
    keep = [i for i in range(len(iw_imgs)) if not leak_mask[i]]
    iw_imgs = [iw_imgs[i] for i in keep]
    iw_lbls = [iw_lbls[i] for i in keep]
    print(f"[leak] transfer set = {len(iw_imgs)} images after exclusions")

    # ---- ground truth, loaded once -----------------------------------------
    iw_gts = [read_yolo_labels(l) for l in iw_lbls]
    hold_gts = [read_yolo_labels(l) for l in hold_lbls]
    n_corn = sum(1 for g in iw_gts for c, _ in g if c == IW_CORN_ID)
    gt_iw_agn = [[(0, b) for c, b in g if c != IW_CORN_ID] for g in iw_gts]
    gt_iw_rag = [[(0, b) for c, b in g if c == IW_RAGWEED_ID] for g in iw_gts]
    gt_ho_agn = [[(0, b) for c, b in g] for g in hold_gts]
    # v3.60.0: holdout GT is filtered by the species' raw cwd12 id, independent
    # of the model's id space.
    gt_ho_rag = [[(0, b) for c, b in g if c == HOLDOUT_ARM_ID] for g in hold_gts]
    hold_species = list(species_names_for(cwd_names))
    if hold_species != CWD12_SPECIES:
        print(f"[sets] WARNING: {REPO / 'cwd12_sealed.yaml'} names {cwd_names} do not "
              f"read as the cwd12 species in id order; the holdout arm still uses "
              f"raw id {HOLDOUT_ARM_ID} as {ARM_SPECIES}")

    results = {"per_seed": {}, "protocol": {
        "conf": CONF, "imgsz": IMGSZ, "iou_nms": IOU_NMS,
        "matcher": "wbf_tta_eval.compute_map (101-pt COCO-style)",
        "iw_class_map_source": "HF dataset card dataset_info, 2026-08-26",
        "corn_gt_boxes_dropped": n_corn,
        "note": ("redrootpigweed (A. retroflexus) deliberately NOT mapped to "
                 "PalmerAmaranth (A. palmeri): same genus, different species"),
        "species_arm": {
            "species": ARM_SPECIES,
            "imageweeds_id": IW_RAGWEED_ID,
            "holdout_cwd12_id": HOLDOUT_ARM_ID,
            "model_class": ("resolved per checkpoint by species "
                            "(species_names_for(model.names)); predictions are "
                            "compared in cwd12 id space"),
            "assumption": ("ImageWeeds 'ragweed' is taken to be common ragweed "
                           "(Ambrosia artemisiifolia); the dataset card gives "
                           "only the common name"),
        }}, "species_arm": {}}

    for seed in SEEDS:
        model = YOLO(str(WEIGHTS[seed]))
        names = model.names
        # The class-agnostic arms are the comparison; the per-species arm needs a
        # head with a Ragweed class. A 100-class merged head has one, a
        # class-agnostic one would not, and that must skip the species arm rather
        # than abort.
        # v3.60.0: the class is found by species, never by the stored name, and
        # predictions are mapped to cwd12 ids so they meet the raw-id holdout GT
        # in one id space.
        to_cwd12 = model_to_cwd12(names)
        rag_ids = sorted(k for k, c in to_cwd12.items() if c == HOLDOUT_ARM_ID)
        has_arm = bool(rag_ids)
        results["species_arm"][seed] = {
            "label_space": label_space(names),
            "model_ids": rag_ids,
            "model_names_as_stored": [names[k] for k in rag_ids],
            "holdout_cwd12_id": HOLDOUT_ARM_ID,
            "imageweeds_id": IW_RAGWEED_ID,
        }
        print(f"[s{seed}] species arm: {results['species_arm'][seed]}", flush=True)
        t0 = time.time()
        preds = {}
        for tag, imgs in (("iw", iw_imgs), ("holdout", hold_imgs)):
            acc = []
            for i, p in enumerate(imgs):
                acc.append(predict_one_scale(model, p, IMGSZ, CONF, IOU_NMS))
                if (i + 1) % 500 == 0:
                    print(f"[s{seed}] {tag} {i+1}/{len(imgs)} "
                          f"({time.time()-t0:.0f}s)", flush=True)
            preds[tag] = acc

        def agn(pr):
            return [(b, s, [0] * len(c)) for b, s, c in pr]

        def only(pr, cwd12_id):
            """Predictions whose class maps to `cwd12_id`, as one class."""
            out = []
            for b, s, c in pr:
                k = [j for j, cc in enumerate(c)
                     if to_cwd12.get(int(cc)) == cwd12_id]
                out.append(([b[j] for j in k], [s[j] for j in k], [0] * len(k)))
            return out

        row = {
            "iw_class_agnostic": compute_map(agn(preds["iw"]), gt_iw_agn, 1,
                                             ["weed"])["mAP50_95"],
            "iw_ragweed_species": (compute_map(only(preds["iw"], HOLDOUT_ARM_ID),
                                               gt_iw_rag, 1, [ARM_SPECIES])["mAP50_95"]
                                   if has_arm else None),
            "holdout_class_agnostic": compute_map(agn(preds["holdout"]),
                                                  gt_ho_agn, 1,
                                                  ["weed"])["mAP50_95"],
            "holdout_ragweed_species": (compute_map(only(preds["holdout"],
                                                         HOLDOUT_ARM_ID),
                                                    gt_ho_rag, 1,
                                                    [ARM_SPECIES])["mAP50_95"]
                                        if has_arm else None),
        }
        results["per_seed"][seed] = row
        print(f"[s{seed}] {json.dumps(row)}", flush=True)

    keys = list(next(iter(results["per_seed"].values())).keys())
    results["summary"] = {}
    for k in keys:
        vals = [results["per_seed"][s][k] for s in SEEDS
                if results["per_seed"][s][k] is not None]
        if vals:
            results["summary"][k] = {"mean": float(np.mean(vals)),
                                     "std": float(np.std(vals)), "n": len(vals)}
    results["leak_check"] = leak
    results["n_transfer_images"] = len(iw_imgs)
    out = output_path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print("\n=== CROSS-DATASET TRANSFER (mean±std, n=3) ===")
    for k in keys:
        if k not in results["summary"]:
            print(f"  {k:24s} (not applicable to these checkpoints)")
            continue
        sm = results["summary"][k]
        print(f"  {k:24s} {sm['mean']:.4f} ± {sm['std']:.4f}  (n={sm['n']})")
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
