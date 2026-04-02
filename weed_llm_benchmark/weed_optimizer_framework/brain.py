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

    def decide_next_action(self, context_history):
        """Brain looks at accumulated context and decides what to do next.

        Args:
            context_history: list of {"role": "system"|"observation", "content": str}

        Returns:
            {"action": str, "params": dict, "reasoning": str}
        """
        prompt = self._build_agent_prompt(context_history)
        response = self._generate(prompt, max_tokens=512)
        action = self._parse_action(response)
        logger.info(f"[Brain] Action: {action.get('action')} — {action.get('reasoning', '')[:80]}")
        return action

    def _build_agent_prompt(self, context_history):
        """Build the agent-mode prompt with full context history."""
        # Format history
        history_text = ""
        for entry in context_history:
            role = entry["role"]
            content = entry["content"]
            if role == "system":
                history_text += f"\n[SYSTEM] {content}\n"
            elif role == "observation":
                history_text += f"\n[OBSERVATION] {content}\n"

        actions_text = "\n".join(f"  - {name}: {desc}" for name, desc in AGENT_ACTIONS.items())

        return f"""You are the Brain of a weed detection optimization system.
You make decisions ONE STEP AT A TIME. After each action, you see the result and decide next.

RULES:
1. ONLY YOLO gets fine-tuned. VLMs are read-only tools.
2. Old species F1 must stay above 0.90.
3. Respect all lessons in the context below.
4. Think carefully before each action — you can inspect, adjust, then train.

AVAILABLE ACTIONS:
{actions_text}

CONTEXT AND HISTORY:
{history_text}

OUTPUT FORMAT — You MUST output valid JSON:
{{"action": "action_name", "params": {{"key": "value"}}, "reasoning": "why this action"}}

Example actions:
{{"action": "inspect_labels", "params": {{"vlm_key": "florence2_base", "sample_size": 20}}, "reasoning": "Check Florence-2 label quality before using them"}}
{{"action": "run_vlm_inference", "params": {{"vlm_key": "owlv2", "max_images": 50}}, "reasoning": "Generate fresh OWLv2 detections on holdout images"}}
{{"action": "generate_consensus", "params": {{"vlm_models": ["florence2_base", "owlv2"], "min_votes": 2, "consensus_iou": 0.3}}, "reasoning": "Create consensus labels from two best VLMs"}}
{{"action": "train_yolo", "params": {{"lr": 0.001, "epochs": 50, "replay_ratio": 0.3}}, "reasoning": "Train YOLO with consensus labels"}}
{{"action": "evaluate", "params": {{}}, "reasoning": "Check if training improved detection"}}
{{"action": "done", "params": {{"reason": "Evaluation shows no improvement, stopping"}}, "reasoning": "No further actions will help"}}

What is your NEXT action? Think step by step, then output JSON:"""

    def _parse_action(self, response):
        """Parse an action from Brain's response."""
        # Try to find JSON
        json_match = re.search(r'\{[^{}]*"action"\s*:\s*"[^"]*"[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                action = json.loads(json_match.group())
                if "action" in action and action["action"] in AGENT_ACTIONS:
                    action.setdefault("params", {})
                    action.setdefault("reasoning", "")
                    return action
            except json.JSONDecodeError:
                pass

        # Code block pattern
        code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if code_match:
            try:
                action = json.loads(code_match.group(1))
                if "action" in action:
                    action.setdefault("params", {})
                    action.setdefault("reasoning", "")
                    return action
            except json.JSONDecodeError:
                pass

        # Fallback: default action sequence
        logger.warning("[Brain] Could not parse action, using fallback")
        return {
            "action": "generate_consensus",
            "params": {"vlm_models": ["florence2_base", "owlv2"], "min_votes": 2, "consensus_iou": 0.3},
            "reasoning": "Fallback: generate consensus with best known pair",
        }

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
