from __future__ import annotations

import os
import uuid

import anthropic

from orchestrator.agents.base import BaseAgentClient

_DEBUG = os.environ.get("ORCHESTRATOR_DEBUG", "").lower() in ("1", "true", "yes")


class AgentClient(BaseAgentClient):
    """Thin wrapper around the Anthropic Managed Agents API.

    Creates a fresh agent + environment + session per `run()` call.
    Suitable for v1 where each run is independent.
    """

    def __init__(
        self,
        name: str,
        model: str,
        system_prompt: str,
        tools: list[dict] | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self._client = anthropic.Anthropic()

    def run(self, user_message: str) -> str:
        """Create a session, send user_message, stream response, return text."""
        # Create a throwaway environment per run (v1 simplicity)
        env = self._client.beta.environments.create(
            name=f"orchestrator-{uuid.uuid4().hex[:8]}",
            config={"type": "cloud", "networking": {"type": "unrestricted"}},
        )

        # Create agent with role-specific config
        agent = self._client.beta.agents.create(
            name=self.name,
            model=self.model,
            system=self.system_prompt,
            tools=self.tools,
        )

        # Create session pinned to this agent version
        session = self._client.beta.sessions.create(
            agent={"type": "agent", "id": agent.id, "version": agent.version},
            environment_id=env.id,
        )

        # Stream-first: open stream, send message, collect response
        text_parts: list[str] = []
        session_errors: list[str] = []

        with self._client.beta.sessions.events.stream(session.id) as stream:
            self._client.beta.sessions.events.send(
                session_id=session.id,
                events=[
                    {
                        "type": "user.message",
                        "content": [{"type": "text", "text": user_message}],
                    }
                ],
            )

            for event in stream:
                if _DEBUG:
                    print(f"[DEBUG] event.type={event.type!r}", flush=True)
                if event.type == "agent.message":
                    if _DEBUG:
                        print(f"[DEBUG]   content blocks: {[b.type for b in event.content]}", flush=True)
                    for block in event.content:
                        if block.type == "text":
                            text_parts.append(block.text)
                elif event.type == "session.error":
                    msg = getattr(event.error, "message", str(event.error))
                    if _DEBUG:
                        print(f"[DEBUG]   session.error: {msg!r}", flush=True)
                    session_errors.append(msg)
                elif event.type == "session.status_idle":
                    if _DEBUG:
                        print(f"[DEBUG]   stop_reason.type={event.stop_reason.type!r}", flush=True)
                    if event.stop_reason.type != "requires_action":
                        break
                elif event.type == "session.status_terminated":
                    break

        result = "".join(text_parts)
        if not result:
            if session_errors:
                raise RuntimeError(
                    f"Session {session.id!r} failed with error: {session_errors[0]}"
                )
            raise RuntimeError(
                f"AgentClient.run() received no text from session {session.id!r}"
            )
        return result
