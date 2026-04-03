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
        """Use Ollama's native function calling to pick a tool."""
        import ollama

        # Build messages
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
        ]
        for entry in context_history[-6:]:  # keep context manageable
            role = entry["role"]
            content = entry["content"][:500]
            if role == "system":
                messages.append({"role": "user", "content": f"[Context] {content}"})
            elif role == "observation":
                messages.append({"role": "user", "content": f"[Result] {content}"})

        messages.append({"role": "user", "content": "What tool should I call next?"})

        try:
            response = ollama.chat(
                model=self.model_id,
                messages=messages,
                tools=TOOL_DEFINITIONS,
            )

            # Check for tool calls
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
                # Model responded with text instead of tool call
                text = response.message.content or ""
                logger.warning(f"[Brain/Ollama] No tool call, text: {text[:100]}")
                # Try to extract action from text
                return self._parse_text_action(text)

        except Exception as e:
            logger.error(f"[Brain/Ollama] Error: {e}")
            return self._smart_fallback(0)

    def _build_system_prompt(self):
        """Concise system prompt for Ollama."""
        return """You are optimizing a YOLO weed detector. Available tools:
- inspect_labels: check VLM label quality
- run_vlm_inference: run Florence-2 or OWLv2 live on images
- generate_consensus: combine VLM detections (best: florence2_base + owlv2)
- train_yolo: train YOLO with labels (ONLY model that gets fine-tuned)
- evaluate: test on old + new species
- identify_weed: use plant.id API for expert species identification
- search_models: find new weed detection models on HuggingFace
- run_external_model: download and run external model (detr_weed, yolov8s_weed)
- done: finish round

RULES: Only YOLO gets fine-tuned. Old species F1 must stay above 0.90.
Best VLM pair: florence2_base (precision=0.789) + owlv2 (recall=0.943).
You can also use external models (DETR, YOLOv8s) as additional label sources."""

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
        {"action": "inspect_labels", "params": {"vlm_key": "florence2_base", "sample_size": 20},
         "reasoning": "Pipeline step 1: inspect best VLM labels"},
        {"action": "inspect_labels", "params": {"vlm_key": "owlv2", "sample_size": 20},
         "reasoning": "Pipeline step 2: inspect second VLM"},
        {"action": "generate_consensus",
         "params": {"vlm_models": ["florence2_base", "owlv2"], "min_votes": 2, "consensus_iou": 0.3},
         "reasoning": "Pipeline step 3: generate consensus from best pair"},
        {"action": "train_yolo", "params": {"lr": 0.001, "epochs": 50, "replay_ratio": 0.3},
         "reasoning": "Pipeline step 4: train YOLO"},
        {"action": "evaluate", "params": {},
         "reasoning": "Pipeline step 5: evaluate"},
        {"action": "done", "params": {"reason": "Pipeline complete"},
         "reasoning": "Pipeline step 6: done"},
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
        for keyword, action_name in [
            ("inspect", "inspect_labels"), ("consensus", "generate_consensus"),
            ("train", "train_yolo"), ("evaluat", "evaluate"),
            ("done", "done"), ("stop", "done"),
        ]:
            if keyword in text_lower:
                return {"action": action_name, "params": {}, "reasoning": f"Text keyword: {keyword}"}
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
