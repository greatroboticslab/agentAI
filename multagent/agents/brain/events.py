"""
Brain-specific event types.
"""

from core.events import Event


class BrainThoughtEvent(Event):
    """Brain completed an analysis cycle."""
    def __init__(self, source: str, summary: dict, thought: str,
                 actions: list, **kwargs):
        super().__init__(
            event_type="brain_thought",
            source=source,
            data={
                "summary": summary,
                "thought": thought,
                "actions": actions,
            },
            **kwargs,
        )


class BrainResponseEvent(Event):
    """Brain response to user chat."""
    def __init__(self, source: str, reply: str, actions_taken: list = None, **kwargs):
        super().__init__(
            event_type="brain_response",
            source=source,
            data={
                "reply": reply,
                "actions_taken": actions_taken or [],
            },
            **kwargs,
        )


class UserChatEvent(Event):
    """User sends a chat message."""
    def __init__(self, source: str, message: str, **kwargs):
        super().__init__(
            event_type="user_chat",
            source=source,
            data={"message": message},
            **kwargs,
        )
