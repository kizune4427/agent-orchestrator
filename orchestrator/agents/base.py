from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAgentClient(ABC):
    """Abstract interface for all agent backends."""

    model: str  # subclasses must set this

    @abstractmethod
    def run(self, user_message: str) -> str:
        """Send user_message and return the text response."""
        ...
