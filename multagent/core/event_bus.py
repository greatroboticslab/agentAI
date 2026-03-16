"""
Async event bus, the core of Agent inter-communication.
Supports:
  - publish/subscribe pattern (one-to-many)
  - request/response pattern (one-to-one, synchronous wait)
  - Event filtering (by type, by source Agent)
  - Event history recording (for Brain review)
  - Dashboard event stream push
"""

import asyncio
import logging
from collections import deque
from typing import Callable, Deque, Dict, List, Optional

from core.events import Event

logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self, history_size: int = 1000):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._history: Deque[Event] = deque(maxlen=history_size)
        self._dashboard_callback: Optional[Callable] = None
        self._request_handlers: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()

    async def publish(self, event: Event) -> None:
        """
        Publish event to the bus.
        All Agents subscribed to this event type will be notified.
        Also pushes to Dashboard.
        """
        self._history.append(event)

        callbacks = self._subscribers.get(event.event_type, [])
        for callback in callbacks:
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.error(f"EventBus callback error for {event.event_type}: {e}")

        # Push to dashboard
        if self._dashboard_callback:
            try:
                result = self._dashboard_callback(event)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.error(f"Dashboard callback error: {e}")

    def subscribe(self, event_type: str, callback: Callable) -> None:
        """Subscribe to a specific event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)
            logger.debug(f"Subscribed to {event_type}")

    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """Unsubscribe from an event type."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(callback)
            except ValueError:
                pass

    async def request(self, target_agent: str, event: Event, timeout: float = 5.0) -> Optional[Event]:
        """Synchronous request-response pattern (with timeout)."""
        future = asyncio.get_event_loop().create_future()
        request_id = event.event_id

        def on_response(response_event: Event):
            if not future.done():
                future.set_result(response_event)

        response_type = f"response_{request_id}"
        self.subscribe(response_type, on_response)

        event.data["_request_id"] = request_id
        event.data["_target_agent"] = target_agent
        await self.publish(event)

        try:
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Request to {target_agent} timed out after {timeout}s")
            return None
        finally:
            self.unsubscribe(response_type, on_response)

    async def respond(self, request_id: str, response_event: Event) -> None:
        """Send a response to a request."""
        response_type = f"response_{request_id}"
        await self.publish(Event(
            event_type=response_type,
            source=response_event.source,
            data=response_event.data,
        ))

    def get_recent_events(self, event_type: str = None, count: int = 50) -> List[Event]:
        """Get recent events (for Brain analysis)."""
        if event_type is None:
            return list(self._history)[-count:]
        return [e for e in self._history if e.event_type == event_type][-count:]

    def set_dashboard_callback(self, callback: Callable) -> None:
        """Register Dashboard push callback."""
        self._dashboard_callback = callback

    @property
    def history_size(self) -> int:
        return len(self._history)
