#!/usr/bin/env python3
"""Does the deployed detector fire on the frames our own robots recorded?

The cross-dataset wall is measured against another *labelled* corpus. This asks
the same question against the corpus the platform actually holds, which has no
labels at all -- so it cannot measure recall or precision, and does not claim to.
What it measures is FIRING: at the deployment confidence threshold, on frames the
robots recorded while driving crop rows, how often does the detector produce a box
at all? A detector at 0.87 on its own holdout that almost never fires on field
video is a domain gap, and the fire rate is the part of that gap measurable
without a labelling campaign.

The vegetation heuristic exists so the answer cannot be explained away by "there
was nothing to detect": excess green (2G-R-B) over a 96x54 thumbnail, and a frame
counts as vegetated above 0.35. The vegetated subset is reported separately.

Aggregates only. No frame, crop or detection image leaves the machine.

    python3 -m weed_optimizer_framework.tools.field_fire_sweep \
        --uploads ~/weed_llm_benchmark/uploads \
        --weights ~/models/cwd12_yolo11n_s102.pt \
        --out results/framework/s6_field_fire.json
"""
import argparse
import json
import os
import pathlib
import time

from .cwd12_species import is_legacy_label_list, species_names_for

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
VEG_BAR = 0.35
THUMB = (96, 54)


def session_frames(d):
    """Every image under a session, whatever layout it was uploaded in.

    Sessions arrive two ways -- `<slug>/frames/*.jpg` from the live uplink and
    `<slug>/images/frames/*.jpg` from a bulk upload -- and an earlier sweep that
    only knew the first layout silently skipped the largest field drive on the
    platform. Walking the tree is the fix; the cost is one stat per file.
    """
    out = []
    for root, _dirs, files in os.walk(str(d)):
        for f in files:
            if pathlib.Path(f).suffix.lower() in IMG_EXT:
                out.append(os.path.join(root, f))
    return sorted(out)


def camera_of(path):
    """The uplink names dual-camera frames `down_*` / `front_*`; others untagged."""
    name = os.path.basename(path)
    for cam in ("down", "front", "left", "right"):
        if name.startswith(cam + "_"):
            return cam
    return "untagged"


def robot_of(slug):
    low = slug.lower()
    if "lasercar" in low:
        return "lasercar"
    if "241" in low or low.startswith("rl_live") or low.startswith("ul_"):
        return "241robot"
    return "unknown"


def veg_fraction(im):
    """Fraction of thumbnail pixels whose excess green exceeds 0.06."""
    t = im.convert("RGB").resize(THUMB)
    px = t.load()
    hit = 0
    for yy in range(THUMB[1]):
        for xx in range(THUMB[0]):
            r, g, b = px[xx, yy]
            if (2 * g - r - b) / 255.0 > 0.06:
                hit += 1
    return hit / float(THUMB[0] * THUMB[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uploads", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", default="0.25,0.40,0.60")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    from PIL import Image
    from ultralytics import YOLO

    confs = [float(c) for c in args.conf.split(",") if c.strip()]
    model = YOLO(args.weights)
    raw_names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    # v3.60.0: count per species; a legacy-labelled checkpoint is translated as a
    # whole list, a checkpoint that already names species passes through.
    names = species_names_for(dict(raw_names))

    up = pathlib.Path(os.path.expanduser(args.uploads))
    sessions = sorted([d for d in up.iterdir() if d.is_dir()])
    out = {"model": os.path.basename(args.weights), "imgsz": args.imgsz,
           "conf_threshold_swept": confs,
           "class_names": names,
           "checkpoint_labels_legacy": is_legacy_label_list(raw_names),
           "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "vegetation_rule": "excess green 2G-R-B > 0.06 on a %dx%d thumbnail; "
                              "frame counts as vegetated above %.2f"
                              % (THUMB[0], THUMB[1], VEG_BAR),
           "sessions": {}}

    tot = {"sessions": 0, "frames": 0, "vegetated": 0}
    for c in confs:
        tot["frames_firing_at_%.2f" % c] = 0
        tot["detections_at_%.2f" % c] = 0
        tot["veg_frames_firing_at_%.2f" % c] = 0

    for sd in sessions:
        frames = session_frames(sd)
        if not frames:
            continue
        rec = {"robot": robot_of(sd.name), "frames": len(frames), "cams": {}}
        veg_flag = []
        t0 = time.time()
        per_conf = {c: {"detections": 0, "frames_firing": 0, "veg_frames_firing": 0,
                        "species": {}} for c in confs}
        for fp in frames:
            rec["cams"][camera_of(fp)] = rec["cams"].get(camera_of(fp), 0) + 1
            with Image.open(fp) as im:
                v = veg_fraction(im)
            veg_flag.append(v > VEG_BAR)
            # One forward pass at the lowest threshold; the higher thresholds are
            # filters over the same boxes, so the sweep costs one inference.
            res = model.predict(fp, imgsz=args.imgsz, conf=min(confs), verbose=False)[0]
            scores = [float(b.conf) for b in res.boxes] if res.boxes is not None else []
            klass = [int(b.cls) for b in res.boxes] if res.boxes is not None else []
            for c in confs:
                keep = [i for i, sc in enumerate(scores) if sc >= c]
                per_conf[c]["detections"] += len(keep)
                if keep:
                    per_conf[c]["frames_firing"] += 1
                    if veg_flag[-1]:
                        per_conf[c]["veg_frames_firing"] += 1
                    for i in keep:
                        n = names.get(klass[i], str(klass[i]))
                        per_conf[c]["species"][n] = per_conf[c]["species"].get(n, 0) + 1
        dt = time.time() - t0
        rec["ms_per_frame"] = round(1000.0 * dt / len(frames), 1)
        rec["vegetated_frames"] = sum(veg_flag)
        for c in confs:
            rec["at_conf_%.2f" % c] = per_conf[c]
        out["sessions"][sd.name] = rec

        tot["sessions"] += 1
        tot["frames"] += len(frames)
        tot["vegetated"] += rec["vegetated_frames"]
        for c in confs:
            tot["frames_firing_at_%.2f" % c] += per_conf[c]["frames_firing"]
            tot["detections_at_%.2f" % c] += per_conf[c]["detections"]
            tot["veg_frames_firing_at_%.2f" % c] += per_conf[c]["veg_frames_firing"]
        print("%-44s %5d frames  %4d vegetated  fired@%.2f %d"
              % (sd.name, len(frames), rec["vegetated_frames"], min(confs),
                 per_conf[min(confs)]["frames_firing"]), flush=True)

    out["totals"] = tot
    out["note"] = ("Aggregate statistics only; no frame left the machine. No ground "
                   "truth exists for these frames, so this measures FIRING BEHAVIOUR, "
                   "not recall or precision.")
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("\nwrote", p)
    print("TOTAL %d sessions %d frames, %d vegetated; fired on %d frame(s) at %.2f"
          % (tot["sessions"], tot["frames"], tot["vegetated"],
             tot["frames_firing_at_%.2f" % min(confs)], min(confs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
