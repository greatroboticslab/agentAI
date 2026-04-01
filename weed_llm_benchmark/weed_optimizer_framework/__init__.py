"""
Weed Optimizer Framework — An agent system for YOLO optimization.

Architecture: SuperBrain (swappable LLM) + Tools (VLM pool, YOLO trainer, evaluator)
Pattern: while loop with tool calling (Brain → Tools → Evaluate → Brain)

Only YOLO gets fine-tuned. All VLMs are read-only assistants.
"""

__version__ = "1.0.0"

from .config import Config
from .memory import Memory
from .monitor import QualityMonitor
from .brain import SuperBrain
from .orchestrator import Orchestrator
