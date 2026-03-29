#!/usr/bin/env python3
"""
HyperAgent: Self-Improving Detection Optimization System

A real LLM (Qwen2.5-7B) serves as the "Brain" that:
1. Analyzes experiment history (all previous strategies + results)
2. Reasons about WHY some strategies work and others don't
3. Proposes new strategies as structured JSON
4. System executes the strategy (generate labels → train YOLO → evaluate)
5. Results fed back to Brain → iterate until convergence

The Brain LLM is modular — can be swapped for any open-source model.

Architecture:
  BrainLLM (Qwen2.5-7B) ← sees full history
      ↓ proposes strategy JSON
  LabelGenerator (multi-VLM consensus)
      ↓
  YOLOTrainer (fine-tune with anti-forgetting)
      ↓
  Evaluator (mAP@0.5, mAP@0.5:0.95, F1 on old+new)
      ↓ results
  BrainLLM ← analyzes, proposes better strategy
      ↓ repeat
"""
import json, os, shutil, time, random, re, gc
import numpy as np
from pathlib import Path
from copy import deepcopy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
L4O_DIR = os.path.join(BASE_DIR, "results", "leave4out")
LABELED_DIR = os.path.join(BASE_DIR, "llm_labeled")
AGENT_DIR = os.path.join(BASE_DIR, "results", "hyperagent")
HF_CACHE = os.environ.get("HF_HOME", "/ocean/projects/cis240145p/byler/hf_cache")

os.makedirs(AGENT_DIR, exist_ok=True)

ALL_CLASSES = {
    0: "Carpetweeds", 1: "Crabgrass", 2: "Eclipta", 3: "Goosegrass",
    4: "Morningglory", 5: "Nutsedge", 6: "PalmerAmaranth", 7: "PricklySida",
    8: "Purslane", 9: "Ragweed", 10: "Sicklepod", 11: "SpottedSpurge",
}
TRAIN_IDS = {0, 1, 6, 7, 8, 9, 10, 11}
SP8_DIR = os.path.join(L4O_DIR, "dataset_8species")
HOLDOUT_DIR = os.path.join(L4O_DIR, "dataset_holdout")
YOLO_8SP = os.path.join(L4O_DIR, "yolo_8species", "train", "weights", "best.pt")

VLM_LABELS = {
    "florence2_base": os.path.join(LABELED_DIR, "florence2_base_cottonweeddet12", "detected", "labels"),
    "florence2": os.path.join(LABELED_DIR, "florence2_cottonweeddet12", "detected", "labels"),
    "owlv2": os.path.join(LABELED_DIR, "owlv2_cottonweeddet12", "detected", "labels"),
    "internvl2": os.path.join(LABELED_DIR, "internvl2_cottonweeddet12", "detected", "labels"),
    "qwen3b": os.path.join(LABELED_DIR, "qwen3b_cottonweeddet12", "detected", "labels"),
    "minicpm_v45": os.path.join(LABELED_DIR, "minicpm_v45_cottonweeddet12", "detected", "labels"),
    "qwen7b": os.path.join(LABELED_DIR, "qwen7b_cottonweeddet12", "detected", "labels"),
}
VLM_PRECISION = {
    "florence2_base": 0.789, "florence2": 0.692, "owlv2": 0.194,
    "internvl2": 0.545, "qwen3b": 0.333, "minicpm_v45": 0.407, "qwen7b": 0.334,
}


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
# Brain LLM (swappable module)
# ============================================================
class BrainLLM:
    """LLM-based strategy proposer. Currently uses Qwen2.5-7B-Instruct."""

    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.history = []

    def load(self):
        """Load LLM to GPU."""
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[*] Loading Brain LLM: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=os.path.join(HF_CACHE, "hub"))
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="auto",
            cache_dir=os.path.join(HF_CACHE, "hub"))
        print(f"[+] Brain loaded")

    def unload(self):
        """Free GPU memory for YOLO training."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            import torch; torch.cuda.empty_cache(); gc.collect()
            print("[*] Brain unloaded, GPU freed")

    def think(self, history):
        """Analyze history and propose next strategy."""
        self.load()

        prompt = self._build_prompt(history)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with __import__('torch').no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=1024, temperature=0.7,
                do_sample=True, top_p=0.9,
            )
        response = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        self.unload()  # Free GPU for YOLO

        # Parse strategy from response
        strategy = self._parse_strategy(response)
        return strategy, response

    def _build_prompt(self, history):
        """Build the prompt for the Brain LLM."""
        prompt = """You are an AI research agent optimizing YOLO object detection for unseen weed species.

PROBLEM: YOLO trained on 8 weed species (F1=0.917) drops to F1=0.606 on 4 unseen species.
We use VLM pseudo-labels to help YOLO learn new species, but this causes catastrophic forgetting.

AVAILABLE VLMs for pseudo-labeling (with their known precision on weed detection):
- florence2_base: precision=0.789 (BEST precision)
- owlv2: precision=0.194 but recall=0.918 (BEST recall)
- florence2: precision=0.692
- internvl2: precision=0.545
- minicpm_v45: precision=0.407
- qwen3b: precision=0.333
- qwen7b: precision=0.334

STRATEGY PARAMETERS you can tune:
- vlm_models: which VLMs to use for pseudo-labeling (list of names)
- min_votes: minimum number of VLMs that must agree (1-7)
- consensus_iou: IoU threshold for matching boxes across VLMs (0.1-0.7)
- use_yolo_old: whether to use YOLO to label old species in new images (true/false)
- lr: learning rate for YOLO training (0.0001-0.01)
- epochs: training epochs (20-100)
- freeze_layers: number of YOLO layers to freeze (0-10)
- replay_ratio: ratio of old training data to mix in (0.0-1.0)

CONSTRAINT: Old species F1 must stay above 0.90. Any strategy causing more forgetting is rejected.

"""
        if history:
            prompt += "EXPERIMENT HISTORY:\n"
            for i, h in enumerate(history):
                cfg = h["config"]
                res = h["result"]
                prompt += f"\nIteration {i}: {cfg.get('name', '?')}\n"
                prompt += f"  Config: vlms={cfg.get('vlm_models')}, votes>={cfg.get('min_votes')}, "
                prompt += f"iou={cfg.get('consensus_iou')}, lr={cfg.get('lr')}, "
                prompt += f"replay={cfg.get('replay_ratio')}, freeze={cfg.get('freeze_layers')}\n"
                prompt += f"  Result: old_F1={res['old_f1']:.3f}, new_F1={res['new_f1']:.3f}"
                if res.get('forgetting'):
                    prompt += " [FORGETTING DETECTED]"
                prompt += "\n"

            best = max(history, key=lambda h: h["result"]["new_f1"] if not h["result"].get("forgetting") else 0)
            prompt += f"\nBEST SO FAR: {best['config'].get('name')} with new_F1={best['result']['new_f1']:.3f}\n"

        prompt += """
Based on the history above, propose the NEXT strategy to try. Think step by step:
1. What patterns do you see in what works vs what doesn't?
2. What specific change would most likely improve new species F1?
3. How to minimize forgetting on old species?

Output your strategy as a JSON object with these exact keys:
{"vlm_models": [...], "min_votes": N, "consensus_iou": X, "use_yolo_old": true/false, "lr": X, "epochs": N, "freeze_layers": N, "replay_ratio": X, "name": "descriptive_name", "reasoning": "why this strategy"}
"""
        return prompt

    def _parse_strategy(self, response):
        """Extract JSON strategy from LLM response."""
        # Try to find JSON in response
        json_match = re.search(r'\{[^{}]*"vlm_models"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                strategy = json.loads(json_match.group())
                # Validate required keys
                required = ["vlm_models", "min_votes", "lr", "epochs", "replay_ratio"]
                if all(k in strategy for k in required):
                    # Set defaults for missing optional keys
                    strategy.setdefault("consensus_iou", 0.3)
                    strategy.setdefault("use_yolo_old", True)
                    strategy.setdefault("freeze_layers", 0)
                    strategy.setdefault("name", "brain_proposed")
                    print(f"  Brain proposed: {strategy.get('name')}")
                    print(f"  Reasoning: {strategy.get('reasoning', 'N/A')[:200]}")
                    return strategy
            except json.JSONDecodeError:
                pass

        # Fallback: if LLM output isn't parseable, use a safe default
        print("  [!] Brain output not parseable, using fallback strategy")
        return {
            "vlm_models": ["florence2_base", "owlv2"],
            "min_votes": 2, "consensus_iou": 0.3, "use_yolo_old": True,
            "lr": 0.0008, "epochs": 60, "freeze_layers": 0, "replay_ratio": 0.4,
            "name": "brain_fallback",
            "reasoning": "Fallback: refine best known strategy with slight lr reduction and more replay"
        }


# ============================================================
# Label Generator (same as agent_optimizer but standalone)
# ============================================================
def generate_labels(config, iteration):
    """Generate consensus pseudo-labels."""
    vlm_models = config.get("vlm_models", ["florence2_base", "owlv2"])
    min_votes = config.get("min_votes", 2)
    consensus_iou = config.get("consensus_iou", 0.3)

    holdout_imgs = os.path.join(HOLDOUT_DIR, "train", "images")
    label_dir = os.path.join(AGENT_DIR, f"labels_r{iteration}")
    os.makedirs(label_dir, exist_ok=True)

    # YOLO old-species labels
    from ultralytics import YOLO
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO(YOLO_8SP)

    stats = {"images": 0, "consensus": 0, "yolo_old": 0}

    for img_file in sorted(os.listdir(holdout_imgs)):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        stem = Path(img_file).stem
        stats["images"] += 1

        # YOLO old species
        old_lines = []
        if config.get("use_yolo_old", True):
            results = yolo.predict(os.path.join(holdout_imgs, img_file), conf=0.25, device=device, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    x, y, w, h = box.xywhn[0].tolist()
                    old_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    stats["yolo_old"] += 1

        # VLM consensus
        all_boxes = []
        for vlm in vlm_models:
            vlm_dir = VLM_LABELS.get(vlm)
            if not vlm_dir or not os.path.isdir(vlm_dir): continue
            lbl = os.path.join(vlm_dir, f"{stem}.txt")
            if not os.path.exists(lbl): continue
            with open(lbl) as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) >= 5:
                        all_boxes.append((float(p[1]), float(p[2]), float(p[3]), float(p[4]), vlm))

        used = set()
        new_lines = []
        for i, bi in enumerate(all_boxes):
            if i in used: continue
            cluster = [i]; used.add(i)
            for j, bj in enumerate(all_boxes):
                if j in used: continue
                if compute_iou(bi[:4], bj[:4]) >= consensus_iou:
                    cluster.append(j); used.add(j)
            vlms = set(all_boxes[k][4] for k in cluster)
            if len(vlms) >= min_votes:
                best = max(cluster, key=lambda k: VLM_PRECISION.get(all_boxes[k][4], 0))
                b = all_boxes[best]
                new_lines.append(f"8 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
                stats["consensus"] += 1

        merged = old_lines + new_lines
        with open(os.path.join(label_dir, f"{stem}.txt"), "w") as f:
            f.write("\n".join(merged) + "\n" if merged else "")

    del yolo; torch.cuda.empty_cache(); gc.collect()
    print(f"  Labels: {stats['images']} imgs, {stats['consensus']} consensus, {stats['yolo_old']} old")
    return label_dir


# ============================================================
# YOLO Trainer
# ============================================================
def train_yolo(config, label_dir, iteration):
    """Train YOLO with given config."""
    from ultralytics import YOLO
    import torch

    ds_dir = os.path.join(AGENT_DIR, f"dataset_r{iteration}")
    for sub in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        os.makedirs(os.path.join(ds_dir, sub), exist_ok=True)

    # Replay buffer
    random.seed(42 + iteration)
    holdout_imgs = os.path.join(HOLDOUT_DIR, "train", "images")
    n_new = len([f for f in os.listdir(label_dir) if f.endswith('.txt') and open(os.path.join(label_dir, f)).read().strip()])
    replay_ratio = config.get("replay_ratio", 0.3)
    all_old = [f for f in os.listdir(os.path.join(SP8_DIR, "train/images")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    n_replay = min(int(n_new * replay_ratio / max(1-replay_ratio, 0.01)), len(all_old))

    for f in random.sample(all_old, n_replay):
        shutil.copy2(os.path.join(SP8_DIR, "train/images", f), os.path.join(ds_dir, "train/images", f))
        stem = Path(f).stem
        lbl = os.path.join(SP8_DIR, "train/labels", f"{stem}.txt")
        if os.path.exists(lbl):
            shutil.copy2(lbl, os.path.join(ds_dir, "train/labels", f"{stem}.txt"))

    for f in os.listdir(os.path.join(SP8_DIR, "valid/images")):
        shutil.copy2(os.path.join(SP8_DIR, "valid/images", f), os.path.join(ds_dir, "valid/images", f))
        stem = Path(f).stem
        lbl = os.path.join(SP8_DIR, "valid/labels", f"{stem}.txt")
        if os.path.exists(lbl):
            shutil.copy2(lbl, os.path.join(ds_dir, "valid/labels", f"{stem}.txt"))

    for lbl_file in os.listdir(label_dir):
        if not lbl_file.endswith(".txt"): continue
        stem = lbl_file.replace(".txt", "")
        with open(os.path.join(label_dir, lbl_file)) as f:
            if not f.read().strip(): continue
        for ext in [".jpg", ".jpeg", ".png", ".JPG"]:
            img = os.path.join(holdout_imgs, stem + ext)
            if os.path.exists(img):
                shutil.copy2(img, os.path.join(ds_dir, f"train/images/new_{stem}{ext}"))
                shutil.copy2(os.path.join(label_dir, lbl_file), os.path.join(ds_dir, f"train/labels/new_{stem}.txt"))
                break

    names = [ALL_CLASSES[old] for old in sorted(TRAIN_IDS)] + ["novel_weed"]
    with open(os.path.join(ds_dir, "data.yaml"), "w") as f:
        f.write(f"train: {os.path.join(ds_dir, 'train', 'images')}\n")
        f.write(f"val: {os.path.join(ds_dir, 'valid', 'images')}\n")
        f.write(f"nc: 9\nnames: {names}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(YOLO_8SP)
    model.train(
        data=os.path.join(ds_dir, "data.yaml"),
        epochs=config.get("epochs", 50), batch=-1, device=device,
        project=os.path.join(AGENT_DIR, f"yolo_r{iteration}"),
        name="train", patience=15,
        lr0=config.get("lr", 0.001),
        freeze=config.get("freeze_layers", 0),
        verbose=False,
    )
    del model; torch.cuda.empty_cache(); gc.collect()
    shutil.rmtree(ds_dir, ignore_errors=True)  # Save disk
    return os.path.join(AGENT_DIR, f"yolo_r{iteration}", "train", "weights", "best.pt")


# ============================================================
# Evaluator
# ============================================================
def evaluate(model_path):
    """Evaluate on old + new species, return structured metrics."""
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
        return round(f1, 4)

    old_f1 = _eval(os.path.join(SP8_DIR, "test/images"), os.path.join(SP8_DIR, "test/labels"))
    new_f1 = _eval(os.path.join(HOLDOUT_DIR, "test/images"), os.path.join(HOLDOUT_DIR, "test/labels"))
    del model; torch.cuda.empty_cache(); gc.collect()

    forgetting = old_f1 < 0.90
    print(f"  Result: old_F1={old_f1:.3f} new_F1={new_f1:.3f} {'FORGETTING!' if forgetting else 'OK'}")
    return {"old_f1": old_f1, "new_f1": new_f1, "forgetting": forgetting}


# ============================================================
# Main HyperAgent Loop
# ============================================================
def main():
    print("=" * 70)
    print("HYPERAGENT: Self-Improving Detection Optimization")
    print("Brain: Qwen2.5-7B-Instruct")
    print("=" * 70)

    brain = BrainLLM("Qwen/Qwen2.5-7B-Instruct")
    history = []
    all_results = {"brain_model": brain.model_id, "rounds": []}

    # Seed with known best result from Phase 3E
    seed = {
        "config": {
            "vlm_models": ["florence2_base", "owlv2"], "min_votes": 2,
            "consensus_iou": 0.3, "use_yolo_old": True,
            "lr": 0.001, "epochs": 50, "freeze_layers": 0, "replay_ratio": 0.3,
            "name": "seed_consensus_flo+owl"
        },
        "result": {"old_f1": 0.897, "new_f1": 0.622, "forgetting": False}
    }
    history.append(seed)
    print(f"\nSeed: old_F1=0.897, new_F1=0.622 (from Phase 3E)")

    max_rounds = 3
    for round_num in range(max_rounds):
        print(f"\n{'='*70}")
        print(f"ROUND {round_num + 1}/{max_rounds}")
        print(f"{'='*70}")

        # Brain thinks
        print("\n--- Brain is thinking... ---")
        strategy, reasoning = brain.think(history)
        strategy["iteration"] = len(history)

        print(f"\n--- Executing: {strategy.get('name')} ---")
        print(f"  VLMs: {strategy.get('vlm_models')}, votes>={strategy.get('min_votes')}")
        print(f"  lr={strategy.get('lr')}, epochs={strategy.get('epochs')}, replay={strategy.get('replay_ratio')}")

        try:
            # Execute strategy
            label_dir = generate_labels(strategy, len(history))
            model_path = train_yolo(strategy, label_dir, len(history))
            result = evaluate(model_path)

            # Record
            entry = {"config": strategy, "result": result, "reasoning": reasoning[:500]}
            history.append(entry)
            all_results["rounds"].append(entry)

            # Check improvement
            if result["new_f1"] > seed["result"]["new_f1"] and not result["forgetting"]:
                print(f"  ★ IMPROVEMENT! new_F1: {seed['result']['new_f1']:.3f} → {result['new_f1']:.3f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            history.append({"config": strategy, "result": {"old_f1": 0, "new_f1": 0, "forgetting": True},
                           "reasoning": f"Error: {str(e)}"})

    # Final summary
    print("\n" + "=" * 70)
    print("HYPERAGENT COMPLETE")
    print("=" * 70)

    valid = [h for h in history if not h["result"].get("forgetting") and h["result"]["new_f1"] > 0]
    if valid:
        best = max(valid, key=lambda h: h["result"]["new_f1"])
        print(f"\nBest strategy: {best['config'].get('name')}")
        print(f"  old_F1={best['result']['old_f1']:.3f}, new_F1={best['result']['new_f1']:.3f}")

    print(f"\nAll rounds:")
    for i, h in enumerate(history):
        r = h["result"]
        print(f"  {i}: {h['config'].get('name','?'):<30s} old={r['old_f1']:.3f} new={r['new_f1']:.3f} "
              f"{'FORGET' if r.get('forgetting') else 'ok'}")

    out = os.path.join(AGENT_DIR, "hyperagent_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[+] Saved to {out}")


if __name__ == "__main__":
    main()
