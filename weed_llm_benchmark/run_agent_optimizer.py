#!/usr/bin/env python3
"""
Autonomous Detection Optimization Agent

An OPRO-inspired self-improving agent that autonomously optimizes YOLO
for unseen weed species using VLM pseudo-labels. The agent:
1. Generates labels via multi-VLM consensus (OWLv2 + Florence-2)
2. Trains YOLO with configurable anti-forgetting strategies
3. Evaluates on old + new species
4. Analyzes results and proposes better strategies
5. Iterates until precision improves

Inspired by: OPRO (ICLR 2024), HyperAgents (Meta 2026), AutoML-Agent (ICML 2025)
"""
import json, os, shutil, time, random, itertools
import numpy as np
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
L4O_DIR = os.path.join(BASE_DIR, "results", "leave4out")
LABELED_DIR = os.path.join(BASE_DIR, "llm_labeled")
AGENT_DIR = os.path.join(BASE_DIR, "results", "agent_optimizer")
os.makedirs(AGENT_DIR, exist_ok=True)

ALL_CLASSES = {
    0: "Carpetweeds", 1: "Crabgrass", 2: "Eclipta", 3: "Goosegrass",
    4: "Morningglory", 5: "Nutsedge", 6: "PalmerAmaranth", 7: "PricklySida",
    8: "Purslane", 9: "Ragweed", 10: "Sicklepod", 11: "SpottedSpurge",
}
TRAIN_IDS = {0, 1, 6, 7, 8, 9, 10, 11}
HOLDOUT_IDS = {2, 3, 4, 5}

# Paths
SP8_DIR = os.path.join(L4O_DIR, "dataset_8species")
HOLDOUT_DIR = os.path.join(L4O_DIR, "dataset_holdout")
YOLO_8SP = os.path.join(L4O_DIR, "yolo_8species", "train", "weights", "best.pt")

# Available VLM label directories
VLM_LABELS = {
    "florence2_base": os.path.join(LABELED_DIR, "florence2_base_cottonweeddet12", "detected", "labels"),
    "florence2": os.path.join(LABELED_DIR, "florence2_cottonweeddet12", "detected", "labels"),
    "owlv2": os.path.join(LABELED_DIR, "owlv2_cottonweeddet12", "detected", "labels"),
    "internvl2": os.path.join(LABELED_DIR, "internvl2_cottonweeddet12", "detected", "labels"),
    "qwen3b": os.path.join(LABELED_DIR, "qwen3b_cottonweeddet12", "detected", "labels"),
    "minicpm_v45": os.path.join(LABELED_DIR, "minicpm_v45_cottonweeddet12", "detected", "labels"),
    "qwen7b": os.path.join(LABELED_DIR, "qwen7b_cottonweeddet12", "detected", "labels"),
}

# Known VLM precision on CottonWeedDet12 (for weighted consensus)
VLM_PRECISION = {
    "florence2_base": 0.789, "florence2": 0.692, "owlv2": 0.194,
    "internvl2": 0.545, "qwen3b": 0.333, "minicpm_v45": 0.407, "qwen7b": 0.334,
}

FORGETTING_THRESHOLD = 0.90  # old species F1 must stay above this


def compute_iou(b1, b2):
    x1_1, y1_1 = b1[0]-b1[2]/2, b1[1]-b1[3]/2
    x2_1, y2_1 = b1[0]+b1[2]/2, b1[1]+b1[3]/2
    x1_2, y1_2 = b2[0]-b2[2]/2, b2[1]-b2[3]/2
    x2_2, y2_2 = b2[0]+b2[2]/2, b2[1]+b2[3]/2
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter/union if union > 0 else 0


# ============================================================
# Component 1: Multi-VLM Consensus Label Generator
# ============================================================
def generate_consensus_labels(config):
    """Generate pseudo-labels using multi-VLM consensus filtering.

    Config keys:
        vlm_models: list of VLM names to use
        min_votes: minimum number of VLMs that must agree
        consensus_iou: IoU threshold for matching boxes across VLMs
        use_yolo_old: whether to also use YOLO for old-species labels
    """
    vlm_models = config.get("vlm_models", ["florence2_base", "owlv2"])
    min_votes = config.get("min_votes", 2)
    consensus_iou = config.get("consensus_iou", 0.3)
    use_yolo_old = config.get("use_yolo_old", True)

    holdout_imgs = os.path.join(HOLDOUT_DIR, "train", "images")
    label_dir = os.path.join(AGENT_DIR, f"labels_iter{config['iteration']}")
    os.makedirs(label_dir, exist_ok=True)

    # Load YOLO old-species predictions if needed
    yolo_preds = {}
    if use_yolo_old:
        from ultralytics import YOLO
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = YOLO(YOLO_8SP)
        for img_file in os.listdir(holdout_imgs):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            stem = Path(img_file).stem
            results = model.predict(os.path.join(holdout_imgs, img_file),
                                     conf=0.25, device=device, verbose=False)
            lines = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    x, y, w, h = box.xywhn[0].tolist()
                    lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            yolo_preds[stem] = lines

    stats = {"total_images": 0, "consensus_detections": 0, "yolo_old_detections": 0}

    for img_file in sorted(os.listdir(holdout_imgs)):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        stem = Path(img_file).stem
        stats["total_images"] += 1

        # Collect all VLM detections for this image
        all_boxes = []  # [(cx, cy, w, h, vlm_name), ...]
        for vlm_name in vlm_models:
            vlm_dir = VLM_LABELS.get(vlm_name)
            if not vlm_dir or not os.path.isdir(vlm_dir): continue
            lbl_path = os.path.join(vlm_dir, f"{stem}.txt")
            if not os.path.exists(lbl_path): continue
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        all_boxes.append((cx, cy, w, h, vlm_name))

        # Cluster boxes by spatial overlap → consensus voting
        used = set()
        consensus_boxes = []
        for i, box_i in enumerate(all_boxes):
            if i in used: continue
            cluster = [i]
            used.add(i)
            for j, box_j in enumerate(all_boxes):
                if j in used: continue
                if compute_iou(box_i[:4], box_j[:4]) >= consensus_iou:
                    cluster.append(j)
                    used.add(j)

            # Count unique VLMs in cluster
            vlms_in_cluster = set(all_boxes[k][4] for k in cluster)
            if len(vlms_in_cluster) >= min_votes:
                # Use the box from the highest-precision VLM
                best_idx = max(cluster, key=lambda k: VLM_PRECISION.get(all_boxes[k][4], 0))
                best_box = all_boxes[best_idx]
                consensus_boxes.append(f"8 {best_box[0]:.6f} {best_box[1]:.6f} {best_box[2]:.6f} {best_box[3]:.6f}")
                stats["consensus_detections"] += 1

        # Merge: YOLO old-species + consensus new-species
        merged = []
        if use_yolo_old and stem in yolo_preds:
            merged.extend(yolo_preds[stem])
            stats["yolo_old_detections"] += len(yolo_preds[stem])
        merged.extend(consensus_boxes)

        with open(os.path.join(label_dir, f"{stem}.txt"), "w") as f:
            f.write("\n".join(merged) + "\n" if merged else "")

    print(f"  Labels: {stats['total_images']} images, "
          f"{stats['consensus_detections']} consensus, {stats['yolo_old_detections']} YOLO old")
    return label_dir, stats


# ============================================================
# Component 2: YOLO Training Manager
# ============================================================
def train_yolo(config, label_dir):
    """Train YOLO with given config and labels."""
    from ultralytics import YOLO
    import torch

    lr = config.get("lr", 0.001)
    epochs = config.get("epochs", 50)
    freeze = config.get("freeze_layers", 0)
    replay_ratio = config.get("replay_ratio", 0.3)
    iteration = config["iteration"]

    # Create dataset
    ds_dir = os.path.join(AGENT_DIR, f"dataset_iter{iteration}")
    for sub in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        os.makedirs(os.path.join(ds_dir, sub), exist_ok=True)

    # Copy 8-species training data (replay buffer)
    count_old = 0
    all_old_imgs = []
    for split in ["train", "valid"]:
        src_imgs = os.path.join(SP8_DIR, split, "images")
        src_lbls = os.path.join(SP8_DIR, split, "labels")
        if not os.path.isdir(src_imgs): continue

        if split == "valid":
            # Always copy all validation data
            for f in os.listdir(src_imgs):
                shutil.copy2(os.path.join(src_imgs, f), os.path.join(ds_dir, "valid/images", f))
                stem = Path(f).stem
                lbl = os.path.join(src_lbls, f"{stem}.txt")
                if os.path.exists(lbl):
                    shutil.copy2(lbl, os.path.join(ds_dir, "valid/labels", f"{stem}.txt"))
        else:
            all_old_imgs = [f for f in os.listdir(src_imgs)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Sample replay buffer from old training data
    random.seed(42 + iteration)
    holdout_imgs_dir = os.path.join(HOLDOUT_DIR, "train", "images")
    n_new = len([f for f in os.listdir(label_dir) if f.endswith('.txt') and
                 open(os.path.join(label_dir, f)).read().strip()])
    n_replay = int(n_new * replay_ratio / max(1 - replay_ratio, 0.01))
    n_replay = min(n_replay, len(all_old_imgs))
    replay_imgs = random.sample(all_old_imgs, n_replay) if n_replay > 0 else []

    for f in replay_imgs:
        src_img = os.path.join(SP8_DIR, "train/images", f)
        src_lbl = os.path.join(SP8_DIR, "train/labels", Path(f).stem + ".txt")
        if os.path.exists(src_img) and os.path.exists(src_lbl):
            shutil.copy2(src_img, os.path.join(ds_dir, "train/images", f))
            shutil.copy2(src_lbl, os.path.join(ds_dir, "train/labels", Path(f).stem + ".txt"))
            count_old += 1

    # Add holdout images with consensus labels
    count_new = 0
    for lbl_file in os.listdir(label_dir):
        if not lbl_file.endswith(".txt"): continue
        stem = lbl_file.replace(".txt", "")
        with open(os.path.join(label_dir, lbl_file)) as f:
            if not f.read().strip(): continue
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            img_path = os.path.join(holdout_imgs_dir, stem + ext)
            if os.path.exists(img_path):
                shutil.copy2(img_path, os.path.join(ds_dir, f"train/images/new_{stem}{ext}"))
                shutil.copy2(os.path.join(label_dir, lbl_file),
                             os.path.join(ds_dir, f"train/labels/new_{stem}.txt"))
                count_new += 1
                break

    # data.yaml
    names = [ALL_CLASSES[old] for old in sorted(TRAIN_IDS)] + ["novel_weed"]
    yaml_path = os.path.join(ds_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"train: {os.path.join(ds_dir, 'train', 'images')}\n")
        f.write(f"val: {os.path.join(ds_dir, 'valid', 'images')}\n")
        f.write(f"nc: 9\nnames: {names}\n")

    print(f"  Training: {count_old} old + {count_new} new images, lr={lr}, epochs={epochs}, freeze={freeze}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(YOLO_8SP)
    model.train(
        data=yaml_path, epochs=epochs, batch=-1, device=device,
        project=os.path.join(AGENT_DIR, f"yolo_iter{iteration}"),
        name="train", patience=15, lr0=lr, freeze=freeze, verbose=False,
    )
    return os.path.join(AGENT_DIR, f"yolo_iter{iteration}", "train", "weights", "best.pt")


# ============================================================
# Component 3: Evaluator
# ============================================================
def evaluate(model_path, label=""):
    """Evaluate on both old and new species. Returns dict with metrics."""
    from ultralytics import YOLO
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)

    def _eval(test_imgs, test_lbls):
        tp, fp, fn = 0, 0, 0
        for f in sorted(os.listdir(test_imgs)):
            if not f.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            stem = Path(f).stem
            lp = os.path.join(test_lbls, stem + ".txt")
            if not os.path.exists(lp): continue
            gt = []
            for l in open(lp):
                p = l.strip().split()
                if len(p) >= 5: gt.append((float(p[1]), float(p[2]), float(p[3]), float(p[4])))
            res = model.predict(os.path.join(test_imgs, f), conf=0.25, device=device, verbose=False)
            preds = [(b.xywhn[0].tolist() + [float(b.conf[0])]) for r in res for b in r.boxes]
            preds.sort(key=lambda x: x[4], reverse=True)
            matched = set()
            for pb in preds:
                bi, bg = 0, -1
                for gi, gb in enumerate(gt):
                    if gi in matched: continue
                    iou = compute_iou(pb[:4], gb)
                    if iou > bi: bi, bg = iou, gi
                if bi >= 0.5 and bg >= 0: matched.add(bg); tp += 1
                else: fp += 1
            fn += len(gt) - len(matched)
        prec = tp/(tp+fp) if (tp+fp) > 0 else 0
        rec = tp/(tp+fn) if (tp+fn) > 0 else 0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    old = _eval(os.path.join(SP8_DIR, "test/images"), os.path.join(SP8_DIR, "test/labels"))
    new = _eval(os.path.join(HOLDOUT_DIR, "test/images"), os.path.join(HOLDOUT_DIR, "test/labels"))
    forgetting = old["f1"] < FORGETTING_THRESHOLD

    print(f"  {label}: old_F1={old['f1']:.3f} new_F1={new['f1']:.3f} {'FORGETTING!' if forgetting else 'OK'}")
    return {"old_species": old, "new_species": new, "forgetting": forgetting}


# ============================================================
# Component 4: Strategy Brain (OPRO-inspired)
# ============================================================
class StrategyBrain:
    """Analyzes experiment history and proposes next strategy."""

    def __init__(self):
        self.history = []
        self.best_score = 0
        self.best_config = None

    def score(self, result):
        """Score a result: maximize new_f1, penalize forgetting."""
        new_f1 = result["new_species"]["f1"]
        old_f1 = result["old_species"]["f1"]
        if result["forgetting"]:
            return new_f1 * 0.5  # heavy penalty
        return new_f1 + 0.1 * old_f1  # bonus for keeping old performance

    def propose_strategies(self):
        """Return a list of strategies to try, informed by history."""
        strategies = []

        if len(self.history) == 0:
            # Iteration 0: baseline strategies covering the search space
            strategies = [
                # Strategy 1: Florence+OWLv2 consensus (2 votes)
                {"vlm_models": ["florence2_base", "owlv2"], "min_votes": 2,
                 "consensus_iou": 0.3, "use_yolo_old": True,
                 "lr": 0.001, "epochs": 50, "freeze_layers": 0, "replay_ratio": 0.3,
                 "name": "consensus_flo+owl_2vote"},

                # Strategy 2: Florence-only, high precision
                {"vlm_models": ["florence2_base"], "min_votes": 1,
                 "consensus_iou": 0.3, "use_yolo_old": True,
                 "lr": 0.0005, "epochs": 50, "freeze_layers": 0, "replay_ratio": 0.5,
                 "name": "florence_only_lowlr"},

                # Strategy 3: 3-model consensus (Florence + OWLv2 + InternVL2)
                {"vlm_models": ["florence2_base", "owlv2", "internvl2"], "min_votes": 2,
                 "consensus_iou": 0.3, "use_yolo_old": True,
                 "lr": 0.001, "epochs": 50, "freeze_layers": 0, "replay_ratio": 0.3,
                 "name": "3model_2vote"},

                # Strategy 4: All 7 VLMs, 3-vote consensus
                {"vlm_models": list(VLM_LABELS.keys()), "min_votes": 3,
                 "consensus_iou": 0.3, "use_yolo_old": True,
                 "lr": 0.001, "epochs": 50, "freeze_layers": 0, "replay_ratio": 0.3,
                 "name": "7model_3vote"},

                # Strategy 5: Florence+OWLv2, frozen backbone
                {"vlm_models": ["florence2_base", "owlv2"], "min_votes": 2,
                 "consensus_iou": 0.3, "use_yolo_old": True,
                 "lr": 0.005, "epochs": 50, "freeze_layers": 10, "replay_ratio": 0.3,
                 "name": "consensus_frozen"},
            ]
        else:
            # Analyze history to propose refined strategies
            # Find best non-forgetting result
            valid = [h for h in self.history if not h["result"]["forgetting"]]
            if valid:
                best = max(valid, key=lambda h: self.score(h["result"]))
                base = deepcopy(best["config"])

                # Variation 1: adjust replay ratio
                for rr in [0.2, 0.4, 0.6]:
                    if rr != base.get("replay_ratio"):
                        v = deepcopy(base)
                        v["replay_ratio"] = rr
                        v["name"] = f"refine_replay{int(rr*100)}"
                        strategies.append(v)

                # Variation 2: adjust learning rate
                for lr in [0.0002, 0.0005, 0.002]:
                    if lr != base.get("lr"):
                        v = deepcopy(base)
                        v["lr"] = lr
                        v["name"] = f"refine_lr{lr}"
                        strategies.append(v)

                # Variation 3: adjust consensus IoU
                for iou in [0.2, 0.4, 0.5]:
                    if iou != base.get("consensus_iou"):
                        v = deepcopy(base)
                        v["consensus_iou"] = iou
                        v["name"] = f"refine_iou{iou}"
                        strategies.append(v)

                # Variation 4: more epochs
                v = deepcopy(base)
                v["epochs"] = 100
                v["name"] = "refine_100ep"
                strategies.append(v)

            if not strategies:
                # Fallback: random exploration
                strategies.append({
                    "vlm_models": random.sample(list(VLM_LABELS.keys()), 3),
                    "min_votes": 2, "consensus_iou": 0.3, "use_yolo_old": True,
                    "lr": random.choice([0.0005, 0.001, 0.002]),
                    "epochs": 50, "freeze_layers": 0,
                    "replay_ratio": random.choice([0.2, 0.3, 0.5]),
                    "name": "random_explore",
                })

        return strategies

    def update(self, config, result):
        """Record experiment result."""
        score = self.score(result)
        self.history.append({"config": config, "result": result, "score": score})
        if score > self.best_score and not result["forgetting"]:
            self.best_score = score
            self.best_config = config
            print(f"  ★ New best! score={score:.3f}")


# ============================================================
# Main Agent Loop
# ============================================================
def main():
    print("=" * 70)
    print("AUTONOMOUS DETECTION OPTIMIZATION AGENT")
    print("Inspired by OPRO (ICLR 2024) + HyperAgents (Meta 2026)")
    print("=" * 70)

    brain = StrategyBrain()

    # Baseline evaluation
    print("\n--- Baseline ---")
    baseline = evaluate(YOLO_8SP, "YOLO 8sp baseline")

    max_iterations = 2  # 2 rounds: 5 initial + refinements
    all_results = {"baseline": baseline, "iterations": []}

    for round_num in range(max_iterations):
        strategies = brain.propose_strategies()
        print(f"\n{'='*70}")
        print(f"ROUND {round_num + 1}: {len(strategies)} strategies to try")
        print(f"{'='*70}")

        for i, config in enumerate(strategies):
            iteration = len(brain.history)
            config["iteration"] = iteration
            name = config.get("name", f"strategy_{iteration}")

            print(f"\n--- Strategy {iteration}: {name} ---")
            print(f"  VLMs: {config.get('vlm_models')}, votes≥{config.get('min_votes')}, "
                  f"lr={config.get('lr')}, replay={config.get('replay_ratio')}")

            try:
                # Step 1: Generate labels
                label_dir, label_stats = generate_consensus_labels(config)

                # Step 2: Train YOLO
                model_path = train_yolo(config, label_dir)

                # Step 3: Evaluate
                result = evaluate(model_path, name)

                # Step 4: Record
                brain.update(config, result)
                all_results["iterations"].append({
                    "iteration": iteration, "name": name,
                    "config": {k: v for k, v in config.items() if k != "iteration"},
                    "label_stats": label_stats, "result": result,
                })

                # Clean up dataset to save disk space
                ds_dir = os.path.join(AGENT_DIR, f"dataset_iter{iteration}")
                if os.path.isdir(ds_dir):
                    shutil.rmtree(ds_dir)

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()
                brain.update(config, {"old_species": {"f1": 0}, "new_species": {"f1": 0}, "forgetting": True})

    # Final summary
    print("\n" + "=" * 70)
    print("AGENT OPTIMIZATION COMPLETE")
    print("=" * 70)

    print(f"\nBaseline: old_F1={baseline['old_species']['f1']:.3f} new_F1={baseline['new_species']['f1']:.3f}")

    if brain.best_config:
        best = [h for h in brain.history if h["config"] == brain.best_config][0]
        print(f"Best strategy: {brain.best_config.get('name')}")
        print(f"  old_F1={best['result']['old_species']['f1']:.3f} "
              f"new_F1={best['result']['new_species']['f1']:.3f}")
        print(f"  Δ old: {best['result']['old_species']['f1'] - baseline['old_species']['f1']:+.3f}")
        print(f"  Δ new: {best['result']['new_species']['f1'] - baseline['new_species']['f1']:+.3f}")

    print(f"\nAll {len(brain.history)} strategies tried:")
    print(f"{'#':<4} {'Name':<30} {'Old F1':>8} {'New F1':>8} {'Score':>8} {'Forget':>8}")
    print("-" * 70)
    for i, h in enumerate(brain.history):
        print(f"{i:<4} {h['config'].get('name','?'):<30} "
              f"{h['result']['old_species']['f1']:8.3f} "
              f"{h['result']['new_species']['f1']:8.3f} "
              f"{h['score']:8.3f} "
              f"{'YES' if h['result']['forgetting'] else 'no':>8}")

    # Save
    out_path = os.path.join(AGENT_DIR, "agent_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[+] Results saved to {out_path}")


if __name__ == "__main__":
    main()
