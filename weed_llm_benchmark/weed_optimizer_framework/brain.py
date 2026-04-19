"""
SuperBrain — The orchestrating LLM that decides what tools to call.

Three Brain backends:
1. Ollama (default) — native function calling, best reliability
2. HuggingFace — direct model loading, used when Ollama unavailable
3. Fallback pipeline — predetermined smart sequence, no LLM needed

Ollama function calling is the key: the model outputs structured tool calls
natively (not free-text JSON we have to parse). This is the same mechanism
used in modern AI coding assistants.

GPU memory for Ollama:
  - Ollama manages its own GPU memory
  - Start ollama serve before running framework
  - Model stays loaded between calls (fast)
  - Unload with ollama stop <model> when done
"""

import gc
import json
import re
import os
import logging
from .config import Config

logger = logging.getLogger(__name__)


# Tool definitions for Ollama function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_labels",
            "description": "Check quality of existing VLM labels. Use this to understand label density, box sizes, and noise level before generating consensus.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vlm_key": {"type": "string", "description": "VLM to inspect: florence2_base, owlv2, internvl2_8b, etc."},
                    "sample_size": {"type": "integer", "description": "Number of images to sample (default 20)"},
                },
                "required": ["vlm_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_vlm_inference",
            "description": "Run live VLM inference on holdout images. Use florence2_base (high precision) or owlv2 (high recall).",
            "parameters": {
                "type": "object",
                "properties": {
                    "vlm_key": {"type": "string", "description": "VLM model: florence2_base or owlv2"},
                    "max_images": {"type": "integer", "description": "Max images to process (default 50)"},
                },
                "required": ["vlm_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_consensus",
            "description": "Generate consensus pseudo-labels by combining detections from multiple VLMs. Only boxes agreed by min_votes VLMs are kept.",
            "parameters": {
                "type": "object",
                "properties": {
                    "vlm_models": {"type": "array", "items": {"type": "string"}, "description": "List of VLM keys to use"},
                    "min_votes": {"type": "integer", "description": "Minimum VLMs that must agree (default 2)"},
                    "consensus_iou": {"type": "number", "description": "IoU threshold for box overlap (default 0.3)"},
                },
                "required": ["vlm_models"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "train_yolo",
            "description": "Train YOLO on consensus labels with replay buffer. This is the ONLY model that gets fine-tuned.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lr": {"type": "number", "description": "Learning rate (default 0.001)"},
                    "epochs": {"type": "integer", "description": "Training epochs (default 50)"},
                    "replay_ratio": {"type": "number", "description": "Ratio of old species data to replay (default 0.3, max 0.5)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate",
            "description": "Evaluate YOLO on old species (8) and new species (4). Returns F1, mAP@0.5, mAP@0.5:0.95 for both.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "identify_weed",
            "description": "Use plant.id web API to professionally identify weed species in images. Limited free calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_images": {"type": "integer", "description": "Number of images to identify (default 5, max 10)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_models",
            "description": "Search HuggingFace for weed detection models. Discovers new models beyond our VLM pool.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query (default: weed detection)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_external_model",
            "description": "Download and run an external weed detection model from HuggingFace. Available: detr_weed, yolov8s_weed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_key": {"type": "string", "description": "Model key: detr_weed, detr_deformable_weed, yolov8s_weed"},
                    "max_images": {"type": "integer", "description": "Max images to process (default 50)"},
                },
                "required": ["model_key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_failure",
            "description": "Analyze WHY the last experiment failed or caused forgetting. Think deeply about root causes before trying the next strategy. Use after evaluate shows bad results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "focus": {"type": "string", "description": "What to analyze: label_noise, forgetting, precision, recall"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_datasets",
            "description": "Search for weed detection datasets on HuggingFace and list all available datasets with download status. Use this to find more training data.",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "Search query (default: weed detection)"},
            }},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "download_dataset",
            "description": "Download a weed dataset. Available: weedsense (120K imgs!), deepweeds (17K), crop_weed_research (4K), grass_weeds (2.5K), weed_crop_aerial (1.2K). More data = better model.",
            "parameters": {"type": "object", "properties": {
                "name": {"type": "string", "description": "Dataset key: weedsense, deepweeds, crop_weed_research, grass_weeds, weed_crop_aerial"},
                "max_images": {"type": "integer", "description": "Limit images (null=all)"},
            }, "required": ["name"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "freeze_train",
            "description": "Train YOLO with backbone freezing (Wang 2025 method). Freezes first N layers to preserve old species knowledge while learning new species. Recommended: freeze=10 (proven to keep COCO performance with 0% degradation while adapting to new domain).",
            "parameters": {
                "type": "object",
                "properties": {
                    "freeze_layers": {"type": "integer", "description": "Number of backbone layers to freeze (0=none, 10=Wang 2025, 14=max safe)"},
                    "lr": {"type": "number", "description": "Learning rate (default 0.001)"},
                    "epochs": {"type": "integer", "description": "Training epochs (default 50)"},
                    "replay_ratio": {"type": "number", "description": "Old data replay ratio (default 0.3)"},
                },
                "required": ["freeze_layers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "two_pass_train",
            "description": "TWO-PASS self-training: (1) train YOLO on noisy labels, (2) use trained YOLO to filter labels at high confidence, (3) retrain on cleaned labels with hybrid LoRA. This directly attacks the 27% label noise bottleneck. Most promising method for precision improvement.",
            "parameters": {
                "type": "object",
                "properties": {
                    "epochs": {"type": "integer", "description": "Epochs per pass (default 30)"},
                    "filter_conf": {"type": "number", "description": "Filter confidence threshold (default 0.8)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lora_train",
            "description": "LoRA training (Low-Rank Adaptation). Injects small trainable adapters into YOLO Conv2d layers, freezes original weights. Parameter-efficient: only ~1% of params train. Theory: preserves base knowledge while adapting to new species. Wang Nature 2025 LoRA-Edge style.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lora_rank": {"type": "integer", "description": "LoRA rank (default 64, higher = more capacity for new species)"},
                    "lora_alpha": {"type": "number", "description": "LoRA scaling alpha (default 32.0)"},
                    "lr": {"type": "number", "description": "Learning rate (default 0.0005, low for LoRA stability)"},
                    "epochs": {"type": "integer", "description": "Training epochs (default 50)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "distill_train",
            "description": "Self-distillation training (Teach YOLO to Remember 2025). Old YOLO acts as teacher, new YOLO learns to match teacher's predictions on old species while adding new species. Combats catastrophic forgetting via knowledge preservation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "distill_alpha": {"type": "number", "description": "Distillation loss weight (default 0.5)"},
                    "lr": {"type": "number", "description": "Learning rate (default 0.0005, lower for distillation)"},
                    "epochs": {"type": "integer", "description": "Training epochs (default 50)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_labels",
            "description": "Filter noisy pseudo-labels using YOLO's own high-confidence predictions. Two-pass training: first train on noisy labels, then use YOLO's conf>0.7 predictions to remove false positives, retrain on filtered labels. This directly attacks the 27% FP noise problem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "confidence_threshold": {"type": "number", "description": "YOLO confidence threshold for filtering (default 0.7)"},
                    "label_dir": {"type": "string", "description": "Directory of labels to filter (default: current consensus labels)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "harvest_new_datasets",
            "description": "V3.0 PRIMARY tool. Each call discovers and downloads up to 5 NEW (not already in registry) bbox-labeled weed/crop/plant detection datasets from HuggingFace. Accumulates permanently across runs — over many runs, total grows toward internet-scale (100K+ images). Use this as the FIRST action of every round to keep growing the dataset pool. Safe to call even when nothing new exists (returns 0 new gracefully).",
            "parameters": {"type": "object", "properties": {
                "max_new": {"type": "integer", "description": "Max new datasets this call (default 5)"},
                "max_images_per_ds": {"type": "integer", "description": "Per-dataset image cap (default 30000; raise if disk permits)"},
                "queries": {"type": "array", "items": {"type": "string"}, "description": "Optional HF search query list. Default spans weed/crop/plant/agriculture/pest."},
            }},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "autolabel_pending",
            "description": "V3.0.11 tool. Run OWLv2 to generate pseudo-bbox labels for every dataset currently in `needs_autolabel` state. Classification datasets (plant-village, plant-disease, etc.) become usable for YOLO training. Since the class is KNOWN to be present in the image (classification GT), OWLv2 localization yields cleaner labels than blind VLM consensus (27% FP). Call after harvest_new_datasets when classification sets were added.",
            "parameters": {"type": "object", "properties": {
                "conf_threshold": {"type": "number", "description": "OWLv2 score threshold (default 0.12; lower=more fallback, higher=fewer labels)"},
                "max_images_per_ds": {"type": "integer", "description": "Per-dataset cap for fast iteration (default: all)"},
                "max_datasets": {"type": "integer", "description": "Max datasets to autolabel this call (default: all pending)"},
            }},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "train_yolo_mega",
            "description": "V3.0 PREFERRED. Merge ALL downloaded real-labeled datasets and train the LARGEST YOLO (Config.DETECTION_MODEL). Use this after downloading at least one bbox dataset. Not for pseudo-labels — for real annotations only. Cumulative: each call retrains on everything downloaded so far.",
            "parameters": {"type": "object", "properties": {
                "base_model": {"type": "string", "description": "Override base weights (yolo11x.pt/yolo26x.pt/etc.). Omit for Config default."},
                "epochs": {"type": "integer", "description": "Training epochs (default 100)"},
                "imgsz": {"type": "integer", "description": "Image size (default 640; use 1024 for max accuracy if VRAM allows)"},
                "lr": {"type": "number", "description": "Learning rate (default 0.001)"},
            }},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "End this optimization round. Use when evaluation is complete or when further actions won't help.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "Why stopping"},
                },
            },
        },
    },
]


class SuperBrain:
    """Swappable LLM Brain with multiple backends."""

    def __init__(self, model_id=None, backend="auto"):
        """Initialize Brain.

        Args:
            model_id: Model name. For Ollama: "qwen2.5:7b". For HF: "Qwen/Qwen2.5-7B-Instruct".
            backend: "ollama", "hf", "fallback", or "auto" (try ollama first).
        """
        self.backend = backend
        self._hf_model = None
        self._hf_tokenizer = None

        # Resolve model ID and backend
        if backend == "auto":
            self.backend = self._detect_backend()

        if self.backend == "ollama":
            self.model_id = model_id or "qwen2.5:7b"
        elif self.backend == "hf":
            self.model_id = model_id or Config.BRAIN_MODELS["qwen25_7b"]["hf_id"]
        else:
            self.model_id = "fallback"

        logger.info(f"[Brain] Backend: {self.backend}, Model: {self.model_id}")

    def _detect_backend(self):
        """Auto-detect best available backend."""
        try:
            import ollama
            ollama.list()  # test connection
            logger.info("[Brain] Ollama detected and connected")
            return "ollama"
        except Exception:
            logger.info("[Brain] Ollama not available, trying HF")
            try:
                import transformers
                return "hf"
            except ImportError:
                logger.warning("[Brain] No LLM backend available, using fallback pipeline")
                return "fallback"

    # =========================================================
    # MAIN API — decide_next_action (used by agent orchestrator)
    # =========================================================

    def decide_next_action(self, context_history, step_num=0):
        """Brain decides what tool to call next.

        Routes to the appropriate backend (Ollama/HF/fallback).
        """
        if self.backend == "ollama":
            return self._ollama_decide(context_history)
        elif self.backend == "hf":
            return self._hf_decide(context_history, step_num)
        else:
            return self._smart_fallback(step_num)

    # =========================================================
    # BACKEND 1: OLLAMA — Native function calling
    # =========================================================

    def _ollama_decide(self, context_history):
        """Use Ollama to pick a tool. Tries native function calling first,
        falls back to text-based reasoning if model doesn't support tools."""
        import ollama

        # Build messages
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
        ]
        for entry in context_history[-6:]:
            role = entry["role"]
            content = entry["content"][:500]
            if role == "system":
                messages.append({"role": "user", "content": f"[Context] {content}"})
            elif role == "observation":
                messages.append({"role": "user", "content": f"[Result] {content}"})

        messages.append({"role": "user", "content": "What tool should I call next? Reply with the tool name."})

        try:
            # Try native function calling first
            response = ollama.chat(
                model=self.model_id,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )

            if response.message.tool_calls:
                tc = response.message.tool_calls[0]
                action = {
                    "action": tc.function.name,
                    "params": tc.function.arguments or {},
                    "reasoning": f"Ollama {self.model_id} tool call",
                }
                logger.info(f"[Brain/Ollama] Tool call: {action['action']} {action['params']}")
                return action
            else:
                text = response.message.content or ""
                logger.info(f"[Brain/Ollama] Text response: {text[:100]}")
                return self._parse_text_action(text)

        except Exception as e:
            error_msg = str(e)
            if "does not support tools" in error_msg:
                # Model doesn't support function calling — use text mode
                logger.info(f"[Brain/Ollama] {self.model_id} doesn't support tools, using text mode")
                return self._ollama_text_decide(messages)
            logger.error(f"[Brain/Ollama] Error: {e}")
            return self._smart_fallback(0)

    def _ollama_text_decide(self, messages):
        """Text-based decision for models without function calling (e.g. DeepSeek-R1)."""
        import ollama

        # v3.0 direction: prioritize dataset discovery + mega training
        messages[-1] = {"role": "user", "content": """Pick the next action by number:
18=harvest_new_datasets (PRIMARY)
15=search_datasets 16=download_dataset 17=train_yolo_mega
1=inspect_labels 2=run_vlm 3=consensus 4=train_yolo 5=evaluate
6=search_models 7=run_external_model 8=analyze_failure 9=filter_labels
10=freeze_train 11=distill_train 12=two_pass_train 13=lora_train 14=done

V3.0 PRIORITY (accumulate internet-scale real bbox labels):
  18 harvest_new_datasets → downloads up to 5 NEW bbox datasets this round (weed/crop/plant)
  17 train_yolo_mega → trains latest YOLO on ALL registered bbox data

Recommended sequence: 18 → 17 → 5 → 14
Reply with JUST the number."""}

        try:
            response = ollama.chat(model=self.model_id, messages=messages)
            text = response.message.content or ""
            logger.info(f"[Brain/Ollama/Text] Response: {text[:150]}")

            # Parse number from response (two-digit first)
            two_digit_map = {
                "18": ("harvest_new_datasets", {"max_new": 15, "max_images_per_ds": 50000}),
                "17": ("train_yolo_mega", {"epochs": 100, "imgsz": 640}),
                "16": ("download_dataset", {"name": "weedsense", "max_images": 60000}),
                "15": ("search_datasets", {"query": "weed detection"}),
                "14": ("done", {"reason": "Brain chose to stop"}),
                "13": ("lora_train", {"lora_rank": 64, "lora_alpha": 128.0, "lr": 0.0005, "epochs": 50, "lora_mode": "hybrid"}),
                "12": ("two_pass_train", {"epochs": 30, "filter_conf": 0.8}),
                "11": ("distill_train", {"distill_alpha": 0.5, "lr": 0.0005, "epochs": 50}),
                "10": ("freeze_train", {"freeze_layers": 10, "lr": 0.001, "epochs": 50, "replay_ratio": 0.3}),
            }
            for two_digit, (name, params) in two_digit_map.items():
                if two_digit in text:
                    return {"action": name, "params": params,
                            "reasoning": f"Brain chose {two_digit}: {name}"}

            for char in text:
                if char in "123456789":
                    action_map = {
                        "1": ("inspect_labels", {"vlm_key": "florence2_base", "sample_size": 20}),
                        "2": ("run_vlm_inference", {"vlm_key": "florence2_base", "max_images": 50}),
                        "3": ("generate_consensus", {"vlm_models": ["florence2_base", "owlv2"], "min_votes": 2, "consensus_iou": 0.3}),
                        "4": ("train_yolo", {"lr": 0.001, "epochs": 50, "replay_ratio": 0.3}),
                        "5": ("evaluate", {}),
                        "6": ("search_models", {"query": "weed detection"}),
                        "7": ("run_external_model", {"model_key": "detr_weed", "max_images": 50}),
                        "8": ("analyze_failure", {"focus": "forgetting"}),
                        "9": ("filter_labels", {"confidence_threshold": 0.8}),
                    }
                    name, params = action_map[char]
                    return {"action": name, "params": params,
                            "reasoning": f"Brain chose {char}: {name}"}

            # Try keyword matching on the full text
            return self._parse_text_action(text)

        except Exception as e:
            logger.error(f"[Brain/Ollama/Text] Error: {e}")
            return self._smart_fallback(0)

    def _build_system_prompt(self):
        """Concise system prompt for Ollama."""
        return """You are optimizing a YOLO weed+crop detector. Goal: ACCURACY CEILING (mAP@0.5:0.95). No real-time constraint.

V3.0 PHILOSOPHY — "collect the internet, train the biggest model":
- Every round, FIRST harvest 5 NEW datasets (weed OR crop OR plant OR agriculture bbox).
- Registry is PERMANENT. Datasets accumulate across all runs. Over many runs → 100K+ real labels.
- Only AFTER harvesting should you train.

REQUIRED FIRST ACTION of every round:
  harvest_new_datasets(max_new=15)
  → finds and downloads up to 15 NEW bbox datasets from HF + GitHub + Kaggle + Roboflow Universe
  → v3.0 north-star is 50K+ real bbox images (currently ~9K); harvest aggressively
  → returns 0 if nothing new found (that's fine, move on)

HARD RULES (violate → orchestrator force-overrides):
  * Call harvest_new_datasets EXACTLY ONCE per round. If it returns 0, DO NOT retry — pool
    is exhausted this round; move on.
  * v3.0.11: After harvest, if the observation mentions any datasets with
    annotation=needs_autolabel OR says "X with bboxes" is less than total downloaded
    images, you MUST call autolabel_pending BEFORE train_yolo_mega. Mega will refuse
    to run when needs_autolabel data exists and will auto-redirect you.
  * Correct round shape: harvest → autolabel_pending (if classification added) →
    train_yolo_mega → evaluate → done.
  * Never call the same tool twice in a row unless its observation explicitly tells you to.

AFTER HARVEST (remaining flow):
  autolabel_pending                          # v3.0.11: OWLv2 pseudo-label every
                                             # needs_autolabel dataset (classification →
                                             # bbox). Unlocks the 380K+ plant-classification
                                             # pool that real-bbox search alone can't hit.
  train_yolo_mega(epochs=50, imgsz=640)    # trains yolo26x on real+autolabel union
  evaluate                                   # measures mAP on old + new species
  done                                       # end round

NOTES:
- Total accumulation matters more than any single round. Don't panic if a round adds 0.
- If you want a specific known dataset: download_dataset('name', max_images=N)
- Legacy v2.x tools exist (generate_consensus, two_pass_train, freeze_train, lora_train...) but
  ONLY use them if train_yolo_mega keeps failing. Real labels > pseudo-labels.

AVAILABLE TOOLS:
- harvest_new_datasets: PRIMARY. 5 new datasets per round (auto-dedup, auto-schema-filter).
- search_datasets / download_dataset: for manual control when harvest isn't enough.
- train_yolo_mega: merges all registered bbox datasets; trains Config.DETECTION_MODEL (yolo26x → fallback).
- evaluate / analyze_failure / done: standard tools.
- Legacy: inspect_labels, run_vlm_inference, generate_consensus, train_yolo, freeze_train,
  distill_train, lora_train, two_pass_train, filter_labels, search_models, run_external_model,
  identify_weed."""

    # =========================================================
    # BACKEND 2: HUGGINGFACE — Direct model loading
    # =========================================================

    def _hf_decide(self, context_history, step_num=0):
        """Use HuggingFace model with simplified numbered prompt."""
        # Compact history
        history_text = ""
        for entry in context_history[-4:]:
            content = entry["content"][:300]
            history_text += f"[{entry['role'].upper()}] {content}\n\n"

        prompt = f"""Choose the next action for weed detection optimization.

{history_text}

Actions: 1=inspect_labels 2=run_vlm 3=consensus 4=train 5=evaluate 6=done
Reply with just the number:"""

        response = self._hf_generate(prompt, max_tokens=50)
        action = self._parse_numbered_action(response, step_num)
        return action

    def _hf_generate(self, prompt, max_tokens=256):
        """Generate with HuggingFace model."""
        self._hf_load()
        import torch
        messages = [{"role": "user", "content": prompt}]
        text = self._hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._hf_tokenizer(text, return_tensors="pt").to(self._hf_model.device)
        with torch.no_grad():
            output = self._hf_model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=0.7, do_sample=True, top_p=0.9)
        response = self._hf_tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        self._hf_unload()
        return response

    def _hf_load(self):
        if self._hf_model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        logger.info(f"[Brain/HF] Loading {self.model_id}...")
        cache_dir = os.path.join(Config.HF_CACHE, "hub")
        self._hf_tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=cache_dir, trust_remote_code=True)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="auto",
            cache_dir=cache_dir, trust_remote_code=True)
        logger.info("[Brain/HF] Loaded")

    def _hf_unload(self):
        if self._hf_model is None:
            return
        del self._hf_model; del self._hf_tokenizer
        self._hf_model = None; self._hf_tokenizer = None
        import torch; torch.cuda.empty_cache(); gc.collect()
        logger.info("[Brain/HF] Unloaded")

    # =========================================================
    # BACKEND 3: FALLBACK — Smart predetermined pipeline
    # =========================================================

    FALLBACK_PIPELINE = [
        # v3.0.11: harvest → autolabel pending → train mega on bbox+autolabel union
        {"action": "harvest_new_datasets", "params": {"max_new": 15, "max_images_per_ds": 50000},
         "reasoning": "Pipeline 1: harvest NEW datasets (HF + GitHub + Kaggle autonomous search)"},
        {"action": "autolabel_pending", "params": {"conf_threshold": 0.12},
         "reasoning": "Pipeline 2: OWLv2 pseudo-label all needs_autolabel datasets (classification → bbox)"},
        {"action": "train_yolo_mega", "params": {"epochs": 50, "imgsz": 640, "patience": 15},
         "reasoning": "Pipeline 3: train yolo26x on union of real bbox + auto-labeled data"},
        {"action": "evaluate", "params": {},
         "reasoning": "Pipeline 4: evaluate mega-trained YOLO"},
        {"action": "done", "params": {"reason": "v3.0 harvest+autolabel+mega cycle complete"},
         "reasoning": "Pipeline 5: done"},
    ]

    def _smart_fallback(self, step_num):
        """Return next action in predetermined pipeline."""
        logger.warning(f"[Brain] Using fallback pipeline (step {step_num})")
        if step_num < len(self.FALLBACK_PIPELINE):
            return dict(self.FALLBACK_PIPELINE[step_num])
        return {"action": "done", "params": {"reason": "Pipeline exhausted"},
                "reasoning": "All fallback steps done"}

    # =========================================================
    # PARSING HELPERS
    # =========================================================

    def _parse_text_action(self, text):
        """Parse action from free-text response (fallback for Ollama no-tool-call)."""
        text_lower = text.lower()
        # Order matters — match most specific first (v3.0 tools before legacy)
        KEYWORD_TABLE = [
            ("harvest_new_datasets", "harvest_new_datasets", {"max_new": 15, "max_images_per_ds": 50000}),
            ("harvest", "harvest_new_datasets", {"max_new": 15, "max_images_per_ds": 50000}),
            ("autolabel_pending", "autolabel_pending", {"conf_threshold": 0.12}),
            ("autolabel", "autolabel_pending", {"conf_threshold": 0.12}),
            ("train_yolo_mega", "train_yolo_mega", {"epochs": 50, "imgsz": 640, "patience": 15}),
            ("search_datasets", "search_datasets", {"query": "weed detection"}),
            ("download_dataset", "download_dataset", {"name": "weedsense", "max_images": 60000}),
            ("mega", "train_yolo_mega", {"epochs": 100, "imgsz": 640}),
            ("search_models", "search_models", {"query": "weed detection"}),
            ("two_pass_train", "two_pass_train", {"epochs": 30, "filter_conf": 0.8}),
            ("two_pass", "two_pass_train", {"epochs": 30, "filter_conf": 0.8}),
            ("lora_train", "lora_train", {"lora_rank": 64, "lora_alpha": 128.0, "lr": 0.0005, "epochs": 50, "lora_mode": "hybrid"}),
            ("distill_train", "distill_train", {"distill_alpha": 0.5, "lr": 0.0005, "epochs": 50}),
            ("freeze_train", "freeze_train", {"freeze_layers": 10, "lr": 0.001, "epochs": 50, "replay_ratio": 0.3}),
            ("filter_labels", "filter_labels", {"confidence_threshold": 0.8}),
            ("analyze_failure", "analyze_failure", {"focus": "forgetting"}),
            ("generate_consensus", "generate_consensus", {"vlm_models": ["florence2_base", "owlv2"], "min_votes": 2, "consensus_iou": 0.3}),
            ("run_vlm", "run_vlm_inference", {"vlm_key": "florence2_base", "max_images": 50}),
            ("inspect_labels", "inspect_labels", {"vlm_key": "florence2_base", "sample_size": 20}),
            ("consensus", "generate_consensus", {"vlm_models": ["florence2_base", "owlv2"], "min_votes": 2, "consensus_iou": 0.3}),
            ("inspect", "inspect_labels", {"vlm_key": "florence2_base", "sample_size": 20}),
            ("evaluat", "evaluate", {}),
            ("download", "download_dataset", {"name": "weedsense", "max_images": 60000}),
            ("dataset", "search_datasets", {"query": "weed detection"}),
            ("train_yolo", "train_yolo", {"lr": 0.001, "epochs": 50, "replay_ratio": 0.3}),
            ("train", "train_yolo_mega", {"epochs": 100, "imgsz": 640}),
            ("done", "done", {"reason": "Brain chose to stop"}),
            ("stop", "done", {"reason": "Brain chose to stop"}),
        ]
        for keyword, action_name, default_params in KEYWORD_TABLE:
            if keyword in text_lower:
                return {"action": action_name, "params": default_params,
                        "reasoning": f"Text keyword: {keyword}"}
        return self._smart_fallback(0)

    def _parse_numbered_action(self, response, step_num=0):
        """Parse numbered action from HF model response."""
        text = response.strip()
        for char in text:
            if char in "123456":
                action_map = {
                    "1": ("inspect_labels", {"vlm_key": "florence2_base", "sample_size": 20}),
                    "2": ("run_vlm_inference", {"vlm_key": "florence2_base", "max_images": 50}),
                    "3": ("generate_consensus", {"vlm_models": ["florence2_base", "owlv2"], "min_votes": 2, "consensus_iou": 0.3}),
                    "4": ("train_yolo", {"lr": 0.001, "epochs": 50, "replay_ratio": 0.3}),
                    "5": ("evaluate", {}),
                    "6": ("done", {"reason": "Brain chose to stop"}),
                }
                name, params = action_map[char]
                return {"action": name, "params": params, "reasoning": f"HF chose option {char}"}
        return self._smart_fallback(step_num)

    # =========================================================
    # STRATEGY MODE + REFLECTION (kept for compatibility)
    # =========================================================

    def analyze_and_propose(self, memory):
        """Strategy mode: propose full strategy JSON."""
        if self.backend == "ollama":
            return self._ollama_strategy(memory)
        elif self.backend == "hf":
            return self._hf_strategy(memory)
        else:
            return Config.DEFAULT_STRATEGY.copy(), "Fallback strategy"

    def _ollama_strategy(self, memory):
        """Use Ollama to propose a strategy."""
        import ollama
        prompt = f"""Propose a weed detection optimization strategy.

{memory.get_summary_for_brain()}

Output JSON: {{"vlm_models": ["florence2_base", "owlv2"], "min_votes": 2,
"consensus_iou": 0.3, "lr": 0.001, "epochs": 50, "replay_ratio": 0.3,
"name": "strategy_name", "reasoning": "why"}}"""

        response = ollama.chat(model=self.model_id,
                               messages=[{"role": "user", "content": prompt}])
        text = response.message.content or ""
        strategy = self._parse_strategy_json(text)
        return strategy, text

    def _hf_strategy(self, memory):
        """Use HF model to propose strategy."""
        prompt = f"""Propose a strategy. {memory.get_summary_for_brain()[:500]}

Output JSON with vlm_models, min_votes, lr, epochs, replay_ratio, name."""
        text = self._hf_generate(prompt)
        strategy = self._parse_strategy_json(text)
        return strategy, text

    def _parse_strategy_json(self, text):
        """Extract strategy JSON from text."""
        json_match = re.search(r'\{[^{}]*"vlm_models"\s*:\s*\[.*?\][^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                s = json.loads(json_match.group())
                if "vlm_models" in s:
                    for k, v in Config.DEFAULT_STRATEGY.items():
                        s.setdefault(k, v)
                    return s
            except json.JSONDecodeError:
                pass
        return dict(Config.DEFAULT_STRATEGY)

    def reflect(self, strategy, result, memory):
        """Generate lesson from experiment result."""
        prompt = f"""Experiment: {json.dumps(strategy, default=str)[:200]}
Result: old_F1={result.get('old_f1', '?')}, new_F1={result.get('new_f1', '?')}, forgetting={result.get('forgetting', '?')}
Baseline: old_F1={memory.baseline.get('old_f1', '?')}, new_F1={memory.baseline.get('new_f1', '?')}
One sentence lesson:"""

        try:
            if self.backend == "ollama":
                import ollama
                r = ollama.chat(model=self.model_id, messages=[{"role": "user", "content": prompt}])
                lesson = (r.message.content or "").strip().split('\n')[0][:200]
            elif self.backend == "hf":
                lesson = self._hf_generate(prompt, max_tokens=100).strip().split('\n')[0][:200]
            else:
                lesson = f"Strategy yielded old_f1={result.get('old_f1')}, new_f1={result.get('new_f1')}"
        except Exception as e:
            lesson = f"Reflection failed: {e}"

        return lesson if len(lesson) > 10 else f"old_f1={result.get('old_f1')}, new_f1={result.get('new_f1')}"

    def unload(self):
        """Unload HF model (Ollama manages its own memory)."""
        self._hf_unload()
