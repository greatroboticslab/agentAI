"""
EmbodiedAction: extends MetaGPT Action with real-time embodied capabilities.
"""

import time
from typing import Any, Optional

from metagpt.actions import Action


class EmbodiedAction(Action):
    """
    Extends MetaGPT Action for embodied AI use cases.
    Adds latency tracking and hardware command support.
    """

    async def run(self, *args, **kwargs) -> Any:
        """
        Run the action with latency tracking.
        Subclasses should override _execute() instead of run().
        """
        start = time.perf_counter()
        try:
            result = await self._execute(*args, **kwargs)
            return result
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._last_latency_ms = elapsed_ms

    async def _execute(self, *args, **kwargs) -> Any:
        """Override this method in subclasses."""
        raise NotImplementedError

    @property
    def last_latency_ms(self) -> float:
        return getattr(self, "_last_latency_ms", 0.0)
