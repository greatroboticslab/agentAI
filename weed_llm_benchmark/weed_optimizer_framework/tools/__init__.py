"""
Tool Registry — Register and dispatch tools that the Brain can call.

Pattern from agent framework: each tool has a name, description, and execute function.
The Brain sees tool descriptions and decides which to call.
"""

import logging
import time
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


class Tool:
    """A single tool that the Brain can invoke."""

    def __init__(self, name: str, func: Callable, description: str,
                 requires_gpu: bool = False):
        self.name = name
        self.func = func
        self.description = description
        self.requires_gpu = requires_gpu
        self.call_count = 0
        self.total_time = 0.0

    def execute(self, **kwargs) -> Any:
        """Execute the tool, tracking timing."""
        logger.info(f"[Tool] Calling {self.name}...")
        start = time.time()
        try:
            result = self.func(**kwargs)
            elapsed = time.time() - start
            self.call_count += 1
            self.total_time += elapsed
            logger.info(f"[Tool] {self.name} completed in {elapsed:.1f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"[Tool] {self.name} failed after {elapsed:.1f}s: {e}")
            raise


class ToolRegistry:
    """Registry of tools available to the Brain.

    Inspired by agent framework's tool system: tools are registered with descriptions,
    and the Brain selects which to call based on its analysis.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, name: str, func: Callable, description: str,
                 requires_gpu: bool = False):
        """Register a tool."""
        self._tools[name] = Tool(name, func, description, requires_gpu)
        logger.debug(f"Registered tool: {name}")

    def call(self, name: str, **kwargs) -> Any:
        """Call a registered tool by name."""
        if name not in self._tools:
            available = list(self._tools.keys())
            raise ValueError(f"Unknown tool '{name}'. Available: {available}")
        return self._tools[name].execute(**kwargs)

    def get_descriptions(self) -> Dict[str, str]:
        """Get all tool descriptions (for Brain context)."""
        return {name: tool.description for name, tool in self._tools.items()}

    def get_descriptions_text(self) -> str:
        """Get tool descriptions as formatted text for Brain prompt."""
        lines = []
        for name, tool in self._tools.items():
            gpu_tag = " [GPU]" if tool.requires_gpu else ""
            lines.append(f"  - {name}{gpu_tag}: {tool.description}")
        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Dict]:
        """Get usage statistics for all tools."""
        return {
            name: {"calls": tool.call_count, "total_time": round(tool.total_time, 1)}
            for name, tool in self._tools.items()
        }

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    @property
    def tool_names(self):
        return list(self._tools.keys())
