"""
SuperBrain — The orchestrating LLM that analyzes, reasons, and proposes strategies.

The Brain sees a context (memory + tool descriptions + history)
and decides what to do next.

Currently: Qwen2.5-7B-Instruct
Future: DeepSeek-R1, Qwen-72B-AWQ, or any stronger open-source reasoning model

The Brain has three operations:
1. analyze_and_propose(): Look at history → propose next strategy
2. reflect(): After an experiment → generate lesson learned
3. diagnose(): When things go wrong → identify root cause

GPU memory management: Brain loads/unloads to share V100-32GB with YOLO.
"""

import gc
import json
import re
import os
import logging
from .config import Config

logger = logging.getLogger(__name__)


class SuperBrain:
    """Swappable LLM that orchestrates YOLO optimization."""

    def __init__(self, model_id=None):
        self.model_id = model_id or Config.BRAIN_MODELS["qwen25_7b"]["hf_id"]
        self.model = None
        self.tokenizer = None
        self._loaded = False

    # --- GPU memory management ---

    def load(self):
        """Load the Brain model onto GPU."""
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
        """Unload Brain from GPU to make room for YOLO."""
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

    # --- Core operations ---

    def analyze_and_propose(self, memory):
        """Main Brain function: analyze experiment history → propose next strategy.

        Args:
            memory: Memory instance with full experiment history

        Returns:
            (strategy_dict, raw_reasoning_text)
        """
        prompt = self._build_analysis_prompt(memory)
        response = self._generate(prompt, max_tokens=1024)
        strategy = self._parse_strategy(response)
        logger.info(f"[Brain] Proposed strategy: {strategy.get('name', '?')}")
        return strategy, response

    def reflect(self, strategy, result, memory):
        """Brain reflects on an experiment outcome → generates lesson.

        Args:
            strategy: the strategy that was executed
            result: the evaluation result
            memory: Memory instance for context

        Returns:
            lesson string
        """
        prompt = self._build_reflect_prompt(strategy, result, memory)
        response = self._generate(prompt, max_tokens=256)
        # Extract first meaningful sentence
        lesson = response.strip().split('\n')[0][:200]
        if len(lesson) < 10:
            lesson = f"Strategy '{strategy.get('name', '?')}' yielded old_f1={result.get('old_f1', '?')}, new_f1={result.get('new_f1', '?')}"
        return lesson

    def diagnose(self, result, memory):
        """Brain diagnoses why a result is bad.

        Args:
            result: the bad evaluation result
            memory: Memory instance for context

        Returns:
            diagnosis string
        """
        prompt = f"""You are an AI research agent diagnosing a failed experiment.

Result: old_F1={result.get('old_f1', '?')}, new_F1={result.get('new_f1', '?')}
Forgetting: {result.get('forgetting', '?')}

Baseline: old_F1={memory.baseline.get('old_f1', '?')}, new_F1={memory.baseline.get('new_f1', '?')}
Best so far: old_F1={memory.current_best.get('old_f1', '?')}, new_F1={memory.current_best.get('new_f1', '?')}

Recent experiments:
{self._format_recent_experiments(memory)}

What is the root cause of this failure? Be specific. Output 2-3 sentences."""

        response = self._generate(prompt, max_tokens=256)
        return response.strip()

    # --- Internal helpers ---

    def _generate(self, prompt, max_tokens=1024):
        """Generate text from the Brain LLM."""
        self.load()
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        response = self.tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        self.unload()
        return response

    def _build_analysis_prompt(self, memory):
        """Build the full context prompt for strategy proposal."""
        context = memory.get_summary_for_brain()

        return f"""You are the SuperBrain of a weed detection optimization system.
Your job: propose a strategy to improve YOLO's detection of UNSEEN weed species
while preventing catastrophic forgetting on KNOWN species.

CRITICAL RULES:
1. ONLY YOLO gets fine-tuned. All VLMs are read-only tools.
2. Old species F1 MUST stay above 0.90.
3. You MUST respect all HARD LESSONS listed below.
4. Think about WHAT HASN'T BEEN TRIED YET — don't repeat failed approaches.

{context}

OUTPUT FORMAT — You must output a valid JSON strategy:
{{
  "vlm_models": ["model1", "model2"],
  "min_votes": 2,
  "consensus_iou": 0.3,
  "use_yolo_old": true,
  "lr": 0.001,
  "epochs": 50,
  "freeze_layers": 0,
  "replay_ratio": 0.3,
  "batch_size": -1,
  "patience": 15,
  "name": "descriptive_strategy_name",
  "reasoning": "1-2 sentences explaining WHY this strategy should work"
}}

Think step by step about what has worked, what has failed, and what to try differently.
Then output your JSON strategy:"""

    def _build_reflect_prompt(self, strategy, result, memory):
        """Build prompt for reflection after an experiment."""
        return f"""You are an AI research agent. An experiment just completed:

Strategy: {json.dumps(strategy, indent=2, default=str)}
Result: old_F1={result.get('old_f1', '?')}, new_F1={result.get('new_f1', '?')}, forgetting={result.get('forgetting', '?')}
mAP50: old={result.get('old_map50', '?')}, new={result.get('new_map50', '?')}
mAP50-95: old={result.get('old_map50_95', '?')}, new={result.get('new_map50_95', '?')}

Baseline: old_F1={memory.baseline.get('old_f1', '?')}, new_F1={memory.baseline.get('new_f1', '?')}
Current best: old_F1={memory.current_best.get('old_f1', '?')}, new_F1={memory.current_best.get('new_f1', '?')}

What is the ONE most important lesson from this experiment?
Be specific, actionable, and concise. Output ONE sentence."""

    def _format_recent_experiments(self, memory):
        """Format recent experiments for prompt context."""
        lines = []
        for e in memory.experiments[-5:]:
            r = e["result"]
            lines.append(f"  {e['strategy'].get('name', '?')}: "
                         f"old={r.get('old_f1', '?')}, new={r.get('new_f1', '?')}")
        return "\n".join(lines) if lines else "  (none)"

    def _parse_strategy(self, response):
        """Parse a strategy JSON from Brain's response.

        Handles common LLM output issues: extra text, markdown code blocks,
        incomplete JSON.
        """
        # Try to find JSON in the response
        # Pattern 1: JSON block within text
        json_match = re.search(
            r'\{[^{}]*"vlm_models"\s*:\s*\[.*?\][^{}]*\}',
            response, re.DOTALL
        )
        if json_match:
            try:
                strategy = json.loads(json_match.group())
                if self._validate_strategy_fields(strategy):
                    return self._fill_defaults(strategy)
            except json.JSONDecodeError:
                pass

        # Pattern 2: code block
        code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if code_match:
            try:
                strategy = json.loads(code_match.group(1))
                if self._validate_strategy_fields(strategy):
                    return self._fill_defaults(strategy)
            except json.JSONDecodeError:
                pass

        # Fallback: slight variation of known best
        logger.warning("[Brain] Could not parse strategy from response, using fallback")
        return {
            "vlm_models": ["florence2_base", "owlv2"],
            "min_votes": 2,
            "consensus_iou": 0.3,
            "use_yolo_old": True,
            "lr": 0.0008,
            "epochs": 60,
            "freeze_layers": 0,
            "replay_ratio": 0.35,
            "batch_size": -1,
            "patience": 15,
            "name": "brain_fallback",
            "reasoning": "Fallback: slight variation of known best strategy",
        }

    def _validate_strategy_fields(self, strategy):
        """Check that a parsed strategy has required fields."""
        required = ["vlm_models", "min_votes"]
        return all(k in strategy for k in required)

    def _fill_defaults(self, strategy):
        """Fill in default values for missing optional fields."""
        defaults = Config.DEFAULT_STRATEGY
        for key, value in defaults.items():
            if key not in strategy:
                strategy[key] = value
        return strategy
