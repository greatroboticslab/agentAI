"""
SuperBrain — The orchestrating LLM that analyzes, reasons, and proposes actions.

Two modes:
1. Strategy mode (original): Brain proposes a full strategy JSON, system executes rigidly
2. Agent mode (new): Brain decides ONE action at a time, sees result, decides next action

Agent mode is the true tool-calling pattern:
    Brain sees context → picks a tool → sees result → picks next tool → ...
    until it decides to stop.

This is fundamentally different from strategy mode because:
- Brain can inspect label quality BEFORE training
- Brain can adjust VLM parameters based on what it sees
- Brain can abort if intermediate results look bad
- Brain learns from each step within a round, not just at the end

GPU memory: Brain loads/unloads to share V100-32GB with YOLO/VLMs.
"""

import gc
import json
import re
import os
import logging
from .config import Config

logger = logging.getLogger(__name__)


# Available actions the Brain can take in agent mode
AGENT_ACTIONS = {
    "inspect_labels": "Check quality of existing VLM labels for a model (vlm_key, sample_size)",
    "run_vlm_inference": "Run live VLM inference on holdout images (vlm_key, max_images)",
    "generate_consensus": "Generate consensus labels from VLM detections (vlm_models, min_votes, consensus_iou)",
    "train_yolo": "Train YOLO with current labels (lr, epochs, replay_ratio)",
    "evaluate": "Evaluate current YOLO on old+new species",
    "done": "Finish this round (with reason)",
}


class SuperBrain:
    """Swappable LLM that orchestrates YOLO optimization."""

    def __init__(self, model_id=None):
        self.model_id = model_id or Config.BRAIN_MODELS["qwen25_7b"]["hf_id"]
        self.model = None
        self.tokenizer = None
        self._loaded = False

    # --- GPU memory management ---

    def load(self):
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"[Brain] Loading {self.model_id}...")
        cache_dir = os.path.join(Config.HF_CACHE, "hub")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, cache_dir=cache_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="auto",
            cache_dir=cache_dir, trust_remote_code=True)
        self._loaded = True
        logger.info("[Brain] Loaded and ready")

    def unload(self):
        if not self._loaded:
            return
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        self._loaded = False
        import torch
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("[Brain] Unloaded, GPU memory freed")

    # =========================================================
    # MODE 2: AGENT MODE — Brain decides one action at a time
    # =========================================================

    def decide_next_action(self, context_history, step_num=0):
        """Brain looks at accumulated context and decides what to do next.

        Uses a simplified numbered-choice format that 7B models can handle.
        If Brain can't decide, uses a smart fallback pipeline that progresses
        through logical steps instead of repeating the same action.

        Args:
            context_history: list of {"role": "system"|"observation", "content": str}
            step_num: current step within the round (for smart fallback)

        Returns:
            {"action": str, "params": dict, "reasoning": str}
        """
        prompt = self._build_agent_prompt(context_history)
        response = self._generate(prompt, max_tokens=256)
        action = self._parse_action(response, step_num)
        logger.info(f"[Brain] Action: {action.get('action')} — {action.get('reasoning', '')[:80]}")
        return action

    def _build_agent_prompt(self, context_history):
        """Simplified prompt — numbered choices for 7B models."""
        # Compact history (last 3 entries only to keep prompt short)
        history_text = ""
        for entry in context_history[-4:]:
            role = entry["role"]
            content = entry["content"][:300]  # truncate long observations
            history_text += f"[{role.upper()}] {content}\n\n"

        return f"""You are an AI agent optimizing weed detection. Choose your next action.

HISTORY:
{history_text}

ACTIONS (pick a number):
1. inspect_labels - Check VLM label quality
2. run_vlm - Run Florence-2 or OWLv2 on images
3. consensus - Generate consensus labels from VLMs
4. train - Train YOLO with labels
5. evaluate - Test YOLO on old+new species
6. done - Stop this round

RESPOND WITH JUST THE NUMBER AND OPTIONAL PARAMETERS.
Example responses:
"1" (inspect Florence-2 labels)
"3" (generate consensus)
"4 lr=0.0005 epochs=40" (train with custom params)
"5" (evaluate)
"6" (done)

Your choice:"""

    def _parse_action(self, response, step_num=0):
        """Parse action from Brain's response. Much more lenient parsing.

        Accepts: "3", "train", '{"action": "train"}', or natural language with keywords.
        Falls back to a smart pipeline that progresses through steps.
        """
        text = response.strip().lower()

        # 1. Try numbered choice (most likely for simplified prompt)
        first_char = text[0] if text else ""
        if first_char in "123456":
            return self._action_from_number(first_char, text)

        # 2. Try JSON (for capable models)
        json_match = re.search(r'\{[^{}]*"action"\s*:\s*"[^"]*"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                action = json.loads(json_match.group())
                if "action" in action and action["action"] in AGENT_ACTIONS:
                    action.setdefault("params", {})
                    action.setdefault("reasoning", "parsed from JSON")
                    return action
            except json.JSONDecodeError:
                pass

        # 3. Try keyword matching
        for keyword, action_name in [
            ("inspect", "inspect_labels"), ("check", "inspect_labels"),
            ("run_vlm", "run_vlm_inference"), ("inference", "run_vlm_inference"),
            ("florence", "run_vlm_inference"), ("owlv2", "run_vlm_inference"),
            ("consensus", "generate_consensus"), ("label", "generate_consensus"),
            ("train", "train_yolo"), ("yolo", "train_yolo"),
            ("evaluat", "evaluate"), ("test", "evaluate"),
            ("done", "done"), ("stop", "done"), ("finish", "done"),
        ]:
            if keyword in text:
                return {"action": action_name, "params": {}, "reasoning": f"Keyword: {keyword}"}

        # 4. Smart fallback pipeline — progresses through steps
        return self._smart_fallback(step_num)

    def _action_from_number(self, num, full_text):
        """Convert numbered choice to action dict."""
        action_map = {
            "1": ("inspect_labels", {"vlm_key": "florence2_base", "sample_size": 20}),
            "2": ("run_vlm_inference", {"vlm_key": "florence2_base", "max_images": 50}),
            "3": ("generate_consensus", {"vlm_models": ["florence2_base", "owlv2"], "min_votes": 2, "consensus_iou": 0.3}),
            "4": ("train_yolo", {"lr": 0.001, "epochs": 50, "replay_ratio": 0.3}),
            "5": ("evaluate", {}),
            "6": ("done", {"reason": "Brain decided to stop"}),
        }
        action_name, default_params = action_map.get(num, ("done", {}))

        # Parse optional params from text like "4 lr=0.0005 epochs=40"
        params = dict(default_params)
        for match in re.finditer(r'(\w+)\s*=\s*([0-9.]+)', full_text):
            key, val = match.group(1), match.group(2)
            try:
                params[key] = float(val) if '.' in val else int(val)
            except ValueError:
                pass

        return {"action": action_name, "params": params, "reasoning": f"Brain chose option {num}"}

    # Smart fallback: predetermined intelligent pipeline
    FALLBACK_PIPELINE = [
        {"action": "inspect_labels", "params": {"vlm_key": "florence2_base", "sample_size": 20},
         "reasoning": "Smart fallback step 1: inspect best VLM labels"},
        {"action": "inspect_labels", "params": {"vlm_key": "owlv2", "sample_size": 20},
         "reasoning": "Smart fallback step 2: inspect second VLM labels"},
        {"action": "generate_consensus", "params": {"vlm_models": ["florence2_base", "owlv2"], "min_votes": 2, "consensus_iou": 0.3},
         "reasoning": "Smart fallback step 3: generate consensus from best pair"},
        {"action": "train_yolo", "params": {"lr": 0.001, "epochs": 50, "replay_ratio": 0.3},
         "reasoning": "Smart fallback step 4: train YOLO"},
        {"action": "evaluate", "params": {},
         "reasoning": "Smart fallback step 5: evaluate results"},
        {"action": "done", "params": {"reason": "Completed full pipeline"},
         "reasoning": "Smart fallback step 6: round complete"},
    ]

    def _smart_fallback(self, step_num):
        """Return the next action in a predetermined pipeline."""
        logger.warning(f"[Brain] Using smart fallback (step {step_num})")
        if step_num < len(self.FALLBACK_PIPELINE):
            return dict(self.FALLBACK_PIPELINE[step_num])
        return {"action": "done", "params": {"reason": "All fallback steps exhausted"},
                "reasoning": "Fallback pipeline complete"}

    # =========================================================
    # MODE 1: STRATEGY MODE — Brain proposes full strategy JSON
    # =========================================================

    def analyze_and_propose(self, memory):
        """Strategy mode: analyze history → propose full strategy JSON."""
        prompt = self._build_strategy_prompt(memory)
        response = self._generate(prompt)
        strategy = self._parse_strategy(response)
        logger.info(f"[Brain] Proposed strategy: {strategy.get('name', '?')}")
        return strategy, response

    def reflect(self, strategy, result, memory):
        """Brain reflects on an experiment outcome → generates lesson."""
        prompt = f"""You are an AI research agent. An experiment just completed:

Strategy: {json.dumps(strategy, indent=2, default=str)}
Result: old_F1={result.get('old_f1', '?')}, new_F1={result.get('new_f1', '?')}, forgetting={result.get('forgetting', '?')}
mAP50: old={result.get('old_map50', '?')}, new={result.get('new_map50', '?')}
mAP50-95: old={result.get('old_map50_95', '?')}, new={result.get('new_map50_95', '?')}

Baseline: old_F1={memory.baseline.get('old_f1', '?')}, new_F1={memory.baseline.get('new_f1', '?')}
Current best: old_F1={memory.current_best.get('old_f1', '?')}, new_F1={memory.current_best.get('new_f1', '?')}

What is the ONE most important lesson from this experiment?
Be specific, actionable, and concise. Output ONE sentence."""

        response = self._generate(prompt, max_tokens=256)
        lesson = response.strip().split('\n')[0][:200]
        if len(lesson) < 10:
            lesson = f"Strategy '{strategy.get('name', '?')}' yielded old_f1={result.get('old_f1', '?')}, new_f1={result.get('new_f1', '?')}"
        return lesson

    def diagnose(self, result, memory):
        """Brain diagnoses why a result is bad."""
        prompt = f"""You are an AI research agent diagnosing a failed experiment.

Result: old_F1={result.get('old_f1', '?')}, new_F1={result.get('new_f1', '?')}
Forgetting: {result.get('forgetting', '?')}
Baseline: old_F1={memory.baseline.get('old_f1', '?')}, new_F1={memory.baseline.get('new_f1', '?')}

What is the root cause? Be specific. Output 2-3 sentences."""

        response = self._generate(prompt, max_tokens=256)
        return response.strip()

    # --- Internal helpers ---

    def _generate(self, prompt, max_tokens=1024):
        self.load()
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=0.7, do_sample=True, top_p=0.9)

        response = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        self.unload()
        return response

    def _build_strategy_prompt(self, memory):
        context = memory.get_summary_for_brain()
        return f"""You are the SuperBrain of a weed detection optimization system.
Your job: propose a strategy to improve YOLO's detection of UNSEEN weed species
while preventing catastrophic forgetting on KNOWN species.

CRITICAL RULES:
1. ONLY YOLO gets fine-tuned. All VLMs are read-only tools.
2. Old species F1 MUST stay above 0.90.
3. You MUST respect all HARD LESSONS listed below.
4. Think about WHAT HASN'T BEEN TRIED YET.

{context}

OUTPUT FORMAT — valid JSON:
{{"vlm_models": ["model1", "model2"], "min_votes": 2, "consensus_iou": 0.3,
  "use_yolo_old": true, "lr": 0.001, "epochs": 50, "freeze_layers": 0,
  "replay_ratio": 0.3, "batch_size": -1, "patience": 15,
  "name": "descriptive_name", "reasoning": "why this should work"}}

Think step by step, then output JSON:"""

    def _parse_strategy(self, response):
        json_match = re.search(
            r'\{[^{}]*"vlm_models"\s*:\s*\[.*?\][^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                strategy = json.loads(json_match.group())
                if self._validate_strategy_fields(strategy):
                    return self._fill_defaults(strategy)
            except json.JSONDecodeError:
                pass

        code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if code_match:
            try:
                strategy = json.loads(code_match.group(1))
                if self._validate_strategy_fields(strategy):
                    return self._fill_defaults(strategy)
            except json.JSONDecodeError:
                pass

        logger.warning("[Brain] Could not parse strategy, using fallback")
        return {
            "vlm_models": ["florence2_base", "owlv2"], "min_votes": 2,
            "consensus_iou": 0.3, "use_yolo_old": True,
            "lr": 0.0008, "epochs": 60, "freeze_layers": 0, "replay_ratio": 0.35,
            "batch_size": -1, "patience": 15, "name": "brain_fallback",
            "reasoning": "Fallback: slight variation of known best strategy",
        }

    def _validate_strategy_fields(self, strategy):
        return all(k in strategy for k in ["vlm_models", "min_votes"])

    def _fill_defaults(self, strategy):
        for key, value in Config.DEFAULT_STRATEGY.items():
            if key not in strategy:
                strategy[key] = value
        return strategy
