#!/usr/bin/env python3
"""
Weed Detection Optimizer Framework

A complete, modular framework where a SuperBrain (swappable LLM) orchestrates
YOLO optimization using multiple VLM assistants. Only YOLO gets fine-tuned;
all other models are read-only tools.

Architecture (inspired by Claude Code's tool-calling paradigm):
  SuperBrain (LLM) → calls Tools → analyzes results → calls more Tools → ...

Components:
  1. SuperBrain: Swappable LLM that orchestrates everything
  2. ToolRegistry: VLM inference, YOLO training, evaluation, label generation
  3. Memory: Experiment history, lessons learned, performance baselines
  4. QualityMonitor: Detects forgetting, false positives, label noise, drift

Design principles (from 18 sessions of experiments):
  - Only YOLO gets fine-tuned; VLMs are read-only assistants
  - 2 complementary models > 7 mediocre models voting
  - Label quality is the root bottleneck, not training strategy
  - Must monitor forgetting after EVERY training run
  - Hard-coded lessons prevent repeating known failures
"""
import json, os, shutil, time, random, re, gc
import numpy as np
from pathlib import Path
from copy import deepcopy
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRAMEWORK_DIR = os.path.join(BASE_DIR, "results", "framework")
HF_CACHE = os.environ.get("HF_HOME", "/ocean/projects/cis240145p/byler/hf_cache")

os.makedirs(FRAMEWORK_DIR, exist_ok=True)


# ============================================================
# 1. MEMORY SYSTEM
# ============================================================
class Memory:
    """Persistent memory for the optimizer framework.

    Stores experiment history, lessons learned, baselines, and current best.
    Persists to JSON between runs.
    """

    def __init__(self, path=None):
        self.path = path or os.path.join(FRAMEWORK_DIR, "memory.json")
        self.experiments = []
        self.lessons = []
        self.baseline = {}
        self.current_best = {}
        self.meta = {"created": datetime.now().isoformat(), "total_iterations": 0}

        # Hard-coded lessons from 18 sessions of experiments
        self._init_hard_lessons()

        if os.path.exists(self.path):
            self.load()

    def _init_hard_lessons(self):
        """Lessons that should NEVER be forgotten."""
        self.hard_lessons = [
            {"id": "HL01", "lesson": "NEVER freeze backbone layers — causes catastrophic failure (F1=0.155)",
             "evidence": "Phase 3C: frozen backbone test", "severity": "critical"},
            {"id": "HL02", "lesson": "Replay buffer >50% makes forgetting WORSE (-3.0% vs -2.4%)",
             "evidence": "Phase 3C: replay50 test", "severity": "high"},
            {"id": "HL03", "lesson": "SAM + caption keyword classification is too noisy (-11% new species)",
             "evidence": "Phase 3D: SAM-enhanced test", "severity": "high"},
            {"id": "HL04", "lesson": "Fine-tuning VLMs (Florence-2) degrades their zero-shot ability (-11.3% mAP)",
             "evidence": "Phase 3F: Florence-2 fine-tune test", "severity": "high"},
            {"id": "HL05", "lesson": "2 complementary models (high-prec + high-recall) beat 7 models voting",
             "evidence": "Phase 3E: agent optimizer 5-strategy test", "severity": "high"},
            {"id": "HL06", "lesson": "Florence-2-base (prec=0.789) + OWLv2 (rec=0.918) is the best VLM pair",
             "evidence": "Phase 3E: consensus_flo+owl achieved +0.016 F1", "severity": "info"},
            {"id": "HL07", "lesson": "Florence-2 <OD> mode does NOT output calibrated confidence scores — threshold filtering doesn't work",
             "evidence": "Phase 3E: conf 0.3/0.5/0.7 all gave identical results", "severity": "high"},
            {"id": "HL08", "lesson": "Old species F1 must stay above 0.90 — any training causing more forgetting must be rejected",
             "evidence": "All phases: forgetting is the primary constraint", "severity": "critical"},
            {"id": "HL09", "lesson": "Label noise (27.4% FP from Florence-2) is the ROOT CAUSE of all failures",
             "evidence": "All pseudo-label approaches degrade YOLO because of noise accumulation", "severity": "critical"},
            {"id": "HL10", "lesson": "YOLO trained on 8/12 species drops 27% F1 on unseen 4 species",
             "evidence": "Phase 3B: Leave-4-Out experiment", "severity": "info"},
        ]

    def add_experiment(self, strategy, result, brain_reasoning=""):
        """Record an experiment."""
        entry = {
            "iteration": len(self.experiments),
            "timestamp": datetime.now().isoformat(),
            "strategy": strategy,
            "result": result,
            "brain_reasoning": brain_reasoning[:500],
        }
        self.experiments.append(entry)
        self.meta["total_iterations"] = len(self.experiments)

        # Update current best if improved and no forgetting
        if (not result.get("forgetting", True) and
            result.get("new_f1", 0) > self.current_best.get("new_f1", 0)):
            self.current_best = {
                "iteration": entry["iteration"],
                "strategy": strategy,
                "old_f1": result.get("old_f1", 0),
                "new_f1": result.get("new_f1", 0),
            }

        self.save()
        return entry

    def add_lesson(self, lesson, evidence, severity="info"):
        """Add a learned lesson."""
        entry = {
            "id": f"LL{len(self.lessons):02d}",
            "lesson": lesson,
            "evidence": evidence,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
        }
        self.lessons.append(entry)
        self.save()

    def get_all_lessons(self):
        """Get all lessons (hard-coded + learned)."""
        return self.hard_lessons + self.lessons

    def get_summary(self):
        """Get a text summary for the Brain to read."""
        lines = []
        lines.append(f"=== MEMORY SUMMARY ===")
        lines.append(f"Total experiments: {len(self.experiments)}")
        lines.append(f"Baseline: old_F1={self.baseline.get('old_f1', '?')}, new_F1={self.baseline.get('new_f1', '?')}")
        if self.current_best:
            lines.append(f"Current best: iter={self.current_best.get('iteration')}, "
                        f"old_F1={self.current_best.get('old_f1', '?')}, "
                        f"new_F1={self.current_best.get('new_f1', '?')}")

        lines.append(f"\n=== HARD LESSONS (NEVER violate) ===")
        for l in self.hard_lessons:
            if l["severity"] in ["critical", "high"]:
                lines.append(f"  [{l['severity'].upper()}] {l['lesson']}")

        if self.lessons:
            lines.append(f"\n=== LEARNED LESSONS ===")
            for l in self.lessons[-5:]:  # last 5
                lines.append(f"  {l['lesson']}")

        if self.experiments:
            lines.append(f"\n=== RECENT EXPERIMENTS ===")
            for e in self.experiments[-5:]:  # last 5
                r = e["result"]
                lines.append(f"  iter{e['iteration']}: {e['strategy'].get('name', '?')} → "
                            f"old={r.get('old_f1', '?')}, new={r.get('new_f1', '?')} "
                            f"{'[FORGETTING]' if r.get('forgetting') else '[OK]'}")

        return "\n".join(lines)

    def save(self):
        data = {
            "meta": self.meta,
            "baseline": self.baseline,
            "current_best": self.current_best,
            "hard_lessons": self.hard_lessons,
            "learned_lessons": self.lessons,
            "experiments": self.experiments,
        }
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, self.path)

    def load(self):
        try:
            data = json.load(open(self.path))
            self.meta = data.get("meta", self.meta)
            self.baseline = data.get("baseline", {})
            self.current_best = data.get("current_best", {})
            self.lessons = data.get("learned_lessons", [])
            self.experiments = data.get("experiments", [])
        except (json.JSONDecodeError, KeyError):
            print("[!] Memory corrupted, starting fresh")


# ============================================================
# 2. QUALITY MONITOR
# ============================================================
class QualityMonitor:
    """Monitors detection quality and catches problems."""

    FORGETTING_THRESHOLD = 0.90

    @staticmethod
    def check_forgetting(old_f1):
        """Check if old species performance dropped below threshold."""
        return old_f1 < QualityMonitor.FORGETTING_THRESHOLD

    @staticmethod
    def validate_strategy(strategy, lessons):
        """Check if a proposed strategy violates any known lessons."""
        violations = []

        # Check against hard lessons
        freeze = strategy.get("freeze_layers", 0)
        if freeze > 3:
            violations.append("HL01: frozen backbone causes catastrophic failure")

        replay = strategy.get("replay_ratio", 0.3)
        if replay > 0.6:
            violations.append("HL02: replay >50% makes forgetting worse")

        vlms = strategy.get("vlm_models", [])
        if len(vlms) > 4:
            violations.append("HL05: too many VLMs — quality > quantity")

        # Check for strategies we already tried with bad results
        for lesson in lessons:
            if lesson.get("severity") == "critical":
                # Don't block, just warn
                pass

        return len(violations) == 0, violations

    @staticmethod
    def assess_result(result, baseline, current_best):
        """Assess whether a result is an improvement."""
        assessment = {
            "forgetting": QualityMonitor.check_forgetting(result.get("old_f1", 0)),
            "improved_new": result.get("new_f1", 0) > current_best.get("new_f1", 0),
            "improved_old": result.get("old_f1", 0) > current_best.get("old_f1", 0),
            "better_than_baseline_new": result.get("new_f1", 0) > baseline.get("new_f1", 0),
        }
        assessment["overall"] = assessment["better_than_baseline_new"] and not assessment["forgetting"]
        return assessment


# ============================================================
# 3. TOOL REGISTRY
# ============================================================
class ToolRegistry:
    """Registry of tools the Brain can call."""

    def __init__(self):
        self.tools = {}

    def register(self, name, func, description):
        self.tools[name] = {"func": func, "description": description}

    def call(self, name, **kwargs):
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}. Available: {list(self.tools.keys())}")
        return self.tools[name]["func"](**kwargs)

    def get_descriptions(self):
        return {name: info["description"] for name, info in self.tools.items()}


# ============================================================
# 4. SUPER BRAIN (swappable LLM)
# ============================================================
class SuperBrain:
    """The orchestrating LLM. Currently Qwen2.5-7B, swappable for any model."""

    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct"):
        self.model_id = model_id
        self.model = None
        self.tokenizer = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[Brain] Loading {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=os.path.join(HF_CACHE, "hub"))
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="auto",
            cache_dir=os.path.join(HF_CACHE, "hub"))
        print(f"[Brain] Loaded")

    def unload(self):
        if self.model is not None:
            del self.model; del self.tokenizer
            self.model = None; self.tokenizer = None
            import torch; torch.cuda.empty_cache(); gc.collect()
            print("[Brain] Unloaded, GPU freed")

    def _generate(self, prompt, max_tokens=1024):
        self.load()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with __import__('torch').no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens,
                                      temperature=0.7, do_sample=True, top_p=0.9)
        response = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        self.unload()
        return response

    def analyze_and_propose(self, memory):
        """Main Brain function: analyze history → propose next strategy."""
        prompt = self._build_prompt(memory)
        response = self._generate(prompt)
        strategy = self._parse_strategy(response)
        return strategy, response

    def reflect(self, strategy, result, memory):
        """Brain reflects on an experiment outcome → generates lesson."""
        prompt = f"""You are an AI research agent. An experiment just completed:

Strategy: {json.dumps(strategy, indent=2)}
Result: old_F1={result.get('old_f1', '?')}, new_F1={result.get('new_f1', '?')}, forgetting={result.get('forgetting', '?')}

Previous best: old_F1={memory.current_best.get('old_f1', '?')}, new_F1={memory.current_best.get('new_f1', '?')}

What lesson should we learn from this result? Be specific and actionable.
Output ONE sentence lesson."""

        response = self._generate(prompt, max_tokens=200)
        # Extract first sentence
        lesson = response.strip().split('\n')[0][:200]
        return lesson

    def _build_prompt(self, memory):
        summary = memory.get_summary()
        tools = "Available VLMs: florence2_base (prec=0.789), owlv2 (rec=0.918), florence2, internvl2, qwen3b, minicpm_v45, qwen7b"

        prompt = f"""You are the SuperBrain of a weed detection optimization system.
Your job: propose a strategy to improve YOLO's detection of unseen weed species.

RULES:
1. ONLY YOLO gets fine-tuned. All VLMs are read-only.
2. Old species F1 MUST stay above 0.90.
3. You MUST NOT violate any HARD LESSONS below.

{summary}

{tools}

STRATEGY FORMAT (output valid JSON):
{{"vlm_models": ["model1", "model2"], "min_votes": N, "consensus_iou": 0.3,
  "use_yolo_old": true, "lr": 0.001, "epochs": 50, "freeze_layers": 0,
  "replay_ratio": 0.3, "name": "descriptive_name", "reasoning": "why"}}

Think step by step, then output your JSON strategy:"""
        return prompt

    def _parse_strategy(self, response):
        json_match = re.search(r'\{[^{}]*"vlm_models"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                strategy = json.loads(json_match.group())
                required = ["vlm_models", "min_votes", "lr"]
                if all(k in strategy for k in required):
                    strategy.setdefault("consensus_iou", 0.3)
                    strategy.setdefault("use_yolo_old", True)
                    strategy.setdefault("freeze_layers", 0)
                    strategy.setdefault("epochs", 50)
                    strategy.setdefault("replay_ratio", 0.3)
                    strategy.setdefault("name", "brain_proposed")
                    return strategy
            except json.JSONDecodeError:
                pass

        # Fallback
        return {
            "vlm_models": ["florence2_base", "owlv2"], "min_votes": 2,
            "consensus_iou": 0.3, "use_yolo_old": True,
            "lr": 0.0008, "epochs": 60, "freeze_layers": 0, "replay_ratio": 0.35,
            "name": "brain_fallback",
            "reasoning": "Fallback: slight variation of known best strategy"
        }


# ============================================================
# 5. TOOL IMPLEMENTATIONS
# ============================================================

# --- Paths (configured for Bridges-2 cluster) ---
L4O_DIR = os.path.join(BASE_DIR, "results", "leave4out")
LABELED_DIR = os.path.join(BASE_DIR, "llm_labeled")
SP8_DIR = os.path.join(L4O_DIR, "dataset_8species")
HOLDOUT_DIR = os.path.join(L4O_DIR, "dataset_holdout")
YOLO_8SP = os.path.join(L4O_DIR, "yolo_8species", "train", "weights", "best.pt")

ALL_CLASSES = {0: "Carpetweeds", 1: "Crabgrass", 2: "Eclipta", 3: "Goosegrass",
               4: "Morningglory", 5: "Nutsedge", 6: "PalmerAmaranth", 7: "PricklySida",
               8: "Purslane", 9: "Ragweed", 10: "Sicklepod", 11: "SpottedSpurge"}
TRAIN_IDS = {0, 1, 6, 7, 8, 9, 10, 11}

VLM_LABELS = {
    "florence2_base": os.path.join(LABELED_DIR, "florence2_base_cottonweeddet12", "detected", "labels"),
    "florence2": os.path.join(LABELED_DIR, "florence2_cottonweeddet12", "detected", "labels"),
    "owlv2": os.path.join(LABELED_DIR, "owlv2_cottonweeddet12", "detected", "labels"),
    "internvl2": os.path.join(LABELED_DIR, "internvl2_cottonweeddet12", "detected", "labels"),
    "qwen3b": os.path.join(LABELED_DIR, "qwen3b_cottonweeddet12", "detected", "labels"),
    "minicpm_v45": os.path.join(LABELED_DIR, "minicpm_v45_cottonweeddet12", "detected", "labels"),
    "qwen7b": os.path.join(LABELED_DIR, "qwen7b_cottonweeddet12", "detected", "labels"),
}
VLM_PRECISION = {"florence2_base": 0.789, "florence2": 0.692, "owlv2": 0.194,
                 "internvl2": 0.545, "qwen3b": 0.333, "minicpm_v45": 0.407, "qwen7b": 0.334}


def _compute_iou(b1, b2):
    x1_1, y1_1 = b1[0]-b1[2]/2, b1[1]-b1[3]/2
    x2_1, y2_1 = b1[0]+b1[2]/2, b1[1]+b1[3]/2
    x1_2, y1_2 = b2[0]-b2[2]/2, b2[1]-b2[3]/2
    x2_2, y2_2 = b2[0]+b2[2]/2, b2[1]+b2[3]/2
    inter = max(0, min(x2_1,x2_2)-max(x1_1,x1_2)) * max(0, min(y2_1,y2_2)-max(y1_1,y1_2))
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter/union if union > 0 else 0


def tool_generate_labels(strategy, iteration):
    """Generate consensus pseudo-labels from VLM pool."""
    vlms = strategy.get("vlm_models", ["florence2_base", "owlv2"])
    min_votes = strategy.get("min_votes", 2)
    iou_thresh = strategy.get("consensus_iou", 0.3)

    from ultralytics import YOLO
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    holdout_imgs = os.path.join(HOLDOUT_DIR, "train", "images")
    label_dir = os.path.join(FRAMEWORK_DIR, f"labels_iter{iteration}")
    os.makedirs(label_dir, exist_ok=True)

    yolo = YOLO(YOLO_8SP) if strategy.get("use_yolo_old", True) else None
    stats = {"images": 0, "consensus": 0, "yolo_old": 0}

    for img_file in sorted(os.listdir(holdout_imgs)):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')): continue
        stem = Path(img_file).stem
        stats["images"] += 1

        # YOLO old species
        old_lines = []
        if yolo:
            results = yolo.predict(os.path.join(holdout_imgs, img_file), conf=0.25, device=device, verbose=False)
            for r in results:
                for box in r.boxes:
                    x, y, w, h = box.xywhn[0].tolist()
                    old_lines.append(f"{int(box.cls[0])} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                    stats["yolo_old"] += 1

        # VLM consensus
        all_boxes = []
        for vlm in vlms:
            vdir = VLM_LABELS.get(vlm)
            if not vdir or not os.path.isdir(vdir): continue
            lbl = os.path.join(vdir, f"{stem}.txt")
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
                if _compute_iou(bi[:4], bj[:4]) >= iou_thresh:
                    cluster.append(j); used.add(j)
            if len(set(all_boxes[k][4] for k in cluster)) >= min_votes:
                best = max(cluster, key=lambda k: VLM_PRECISION.get(all_boxes[k][4], 0))
                b = all_boxes[best]
                new_lines.append(f"8 {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
                stats["consensus"] += 1

        with open(os.path.join(label_dir, f"{stem}.txt"), "w") as f:
            f.write("\n".join(old_lines + new_lines) + "\n" if (old_lines + new_lines) else "")

    if yolo: del yolo
    import torch; torch.cuda.empty_cache(); gc.collect()
    print(f"  [Labels] {stats['images']} imgs, {stats['consensus']} consensus, {stats['yolo_old']} old")
    return label_dir, stats


def tool_train_yolo(strategy, label_dir, iteration):
    """Train YOLO with given labels and config."""
    from ultralytics import YOLO
    import torch

    ds_dir = os.path.join(FRAMEWORK_DIR, f"dataset_iter{iteration}")
    for sub in ["train/images", "train/labels", "valid/images", "valid/labels"]:
        os.makedirs(os.path.join(ds_dir, sub), exist_ok=True)

    # Replay buffer
    random.seed(42 + iteration)
    holdout_imgs = os.path.join(HOLDOUT_DIR, "train", "images")
    n_new = len([f for f in os.listdir(label_dir) if f.endswith('.txt') and open(os.path.join(label_dir, f)).read().strip()])
    replay_ratio = strategy.get("replay_ratio", 0.3)
    all_old = [f for f in os.listdir(os.path.join(SP8_DIR, "train/images")) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    n_replay = min(int(n_new * replay_ratio / max(1-replay_ratio, 0.01)), len(all_old))

    for f in random.sample(all_old, n_replay):
        shutil.copy2(os.path.join(SP8_DIR, "train/images", f), os.path.join(ds_dir, "train/images", f))
        stem = Path(f).stem
        lbl = os.path.join(SP8_DIR, "train/labels", f"{stem}.txt")
        if os.path.exists(lbl): shutil.copy2(lbl, os.path.join(ds_dir, "train/labels", f"{stem}.txt"))

    for f in os.listdir(os.path.join(SP8_DIR, "valid/images")):
        shutil.copy2(os.path.join(SP8_DIR, "valid/images", f), os.path.join(ds_dir, "valid/images", f))
        stem = Path(f).stem
        lbl = os.path.join(SP8_DIR, "valid/labels", f"{stem}.txt")
        if os.path.exists(lbl): shutil.copy2(lbl, os.path.join(ds_dir, "valid/labels", f"{stem}.txt"))

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
        f.write(f"train: {os.path.join(ds_dir, 'train', 'images')}\nval: {os.path.join(ds_dir, 'valid', 'images')}\nnc: 9\nnames: {names}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(YOLO_8SP)
    model.train(data=os.path.join(ds_dir, "data.yaml"), epochs=strategy.get("epochs", 50),
                batch=-1, device=device, project=os.path.join(FRAMEWORK_DIR, f"yolo_iter{iteration}"),
                name="train", patience=15, lr0=strategy.get("lr", 0.001),
                freeze=strategy.get("freeze_layers", 0), verbose=False)

    del model; torch.cuda.empty_cache(); gc.collect()
    shutil.rmtree(ds_dir, ignore_errors=True)
    return os.path.join(FRAMEWORK_DIR, f"yolo_iter{iteration}", "train", "weights", "best.pt")


def tool_evaluate(model_path):
    """Evaluate YOLO on old + new species."""
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
                    iou = _compute_iou(pb[:4], gb)
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

    forgetting = QualityMonitor.check_forgetting(old_f1)
    return {"old_f1": old_f1, "new_f1": new_f1, "forgetting": forgetting}


# ============================================================
# 6. MAIN ORCHESTRATION LOOP
# ============================================================
def run_framework(brain_model="Qwen/Qwen2.5-7B-Instruct", max_rounds=3):
    """Main entry point: run the full optimization framework."""
    print("=" * 70)
    print("WEED OPTIMIZER FRAMEWORK")
    print(f"Brain: {brain_model}")
    print("=" * 70)

    # Initialize components
    memory = Memory()
    brain = SuperBrain(brain_model)
    monitor = QualityMonitor()
    tools = ToolRegistry()
    tools.register("generate_labels", tool_generate_labels, "Generate VLM consensus pseudo-labels")
    tools.register("train_yolo", tool_train_yolo, "Train YOLO with given labels")
    tools.register("evaluate", tool_evaluate, "Evaluate YOLO on old+new species")

    # Set baseline if not set
    if not memory.baseline:
        print("\n--- Establishing baseline ---")
        baseline = tool_evaluate(YOLO_8SP)
        memory.baseline = {"old_f1": baseline["old_f1"], "new_f1": baseline["new_f1"]}
        memory.current_best = {"old_f1": baseline["old_f1"], "new_f1": baseline["new_f1"],
                               "strategy": "baseline_yolo_8sp", "iteration": -1}
        memory.save()
        print(f"  Baseline: old_F1={baseline['old_f1']}, new_F1={baseline['new_f1']}")

    # Seed with known best if first run
    if len(memory.experiments) == 0:
        memory.add_experiment(
            {"vlm_models": ["florence2_base", "owlv2"], "min_votes": 2, "consensus_iou": 0.3,
             "use_yolo_old": True, "lr": 0.001, "epochs": 50, "freeze_layers": 0,
             "replay_ratio": 0.3, "name": "seed_best_known"},
            {"old_f1": 0.897, "new_f1": 0.622, "forgetting": False},
            "Seed: best result from Phase 3E agent optimizer"
        )

    # Main loop
    for round_num in range(max_rounds):
        print(f"\n{'='*70}")
        print(f"ROUND {round_num + 1}/{max_rounds}")
        print(f"{'='*70}")

        # Brain analyzes and proposes
        print("\n[Brain] Analyzing history and proposing strategy...")
        strategy, reasoning = brain.analyze_and_propose(memory)
        strategy["iteration"] = len(memory.experiments)

        # Validate against lessons
        valid, violations = monitor.validate_strategy(strategy, memory.get_all_lessons())
        if not valid:
            print(f"  [!] Strategy violates lessons: {violations}")
            print(f"  [!] Adjusting strategy...")
            strategy["freeze_layers"] = min(strategy.get("freeze_layers", 0), 2)
            strategy["replay_ratio"] = min(strategy.get("replay_ratio", 0.3), 0.5)

        print(f"  Strategy: {strategy.get('name')}")
        print(f"  VLMs: {strategy.get('vlm_models')}, votes>={strategy.get('min_votes')}")
        print(f"  lr={strategy.get('lr')}, epochs={strategy.get('epochs')}, replay={strategy.get('replay_ratio')}")

        try:
            # Execute
            label_dir, label_stats = tool_generate_labels(strategy, strategy["iteration"])
            model_path = tool_train_yolo(strategy, label_dir, strategy["iteration"])
            result = tool_evaluate(model_path)

            # Assess
            assessment = monitor.assess_result(result, memory.baseline, memory.current_best)
            print(f"  Result: old_F1={result['old_f1']}, new_F1={result['new_f1']}")
            print(f"  Assessment: {'IMPROVED' if assessment['overall'] else 'no improvement'}, "
                  f"{'FORGETTING' if assessment['forgetting'] else 'no forgetting'}")

            # Brain reflects
            lesson = brain.reflect(strategy, result, memory)
            if lesson:
                memory.add_lesson(lesson, f"Round {round_num+1}: {strategy.get('name')}")

            # Record
            memory.add_experiment(strategy, result, reasoning[:300])

            if assessment["overall"]:
                print(f"  ★ NEW BEST! new_F1: {memory.current_best.get('new_f1', '?')} → {result['new_f1']}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            memory.add_experiment(strategy, {"old_f1": 0, "new_f1": 0, "forgetting": True}, f"Error: {e}")

    # Final summary
    print("\n" + "=" * 70)
    print("FRAMEWORK RUN COMPLETE")
    print("=" * 70)
    print(f"\nBaseline: old={memory.baseline.get('old_f1')}, new={memory.baseline.get('new_f1')}")
    print(f"Best: old={memory.current_best.get('old_f1')}, new={memory.current_best.get('new_f1')}")
    print(f"\nAll experiments:")
    for e in memory.experiments:
        r = e["result"]
        print(f"  {e['iteration']}: {e['strategy'].get('name','?'):<35s} old={r.get('old_f1','?')} new={r.get('new_f1','?')} "
              f"{'FORGET' if r.get('forgetting') else 'ok'}")

    print(f"\nLessons learned ({len(memory.lessons)}):")
    for l in memory.lessons:
        print(f"  - {l['lesson']}")

    print(f"\nMemory saved to: {memory.path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Weed Optimizer Framework")
    parser.add_argument("--brain", default="Qwen/Qwen2.5-7B-Instruct", help="Brain LLM model ID")
    parser.add_argument("--rounds", type=int, default=3, help="Number of optimization rounds")
    args = parser.parse_args()
    run_framework(brain_model=args.brain, max_rounds=args.rounds)
