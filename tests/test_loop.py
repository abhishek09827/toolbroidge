import pytest

from toolbridge.decorator import tool
from toolbridge.loop import Agent, AgentLoopError, ToolNotFoundError


DESCRIPTION = "Use when echoing text. Do not use for changing external systems."


@tool(DESCRIPTION)
def echo(text: str) -> str:
    return text


@tool("Use when saving text. Do not use for read only answers.", consequential=True)
def save_text(text: str) -> str:
    return f"saved:{text}"


class EndlessProvider:
    name = "fake"

    def complete(self, **kwargs):
        return {"tool_calls": [{"id": "1", "name": "echo", "arguments": {"text": "again"}}]}


class UnknownProvider:
    name = "fake"

    def complete(self, **kwargs):
        return {"tool_calls": [{"id": "1", "name": "missing", "arguments": {}}]}


class ConsequentialProvider:
    name = "fake"

    def __init__(self):
        self.turn = 0

    def complete(self, **kwargs):
        self.turn += 1
        if self.turn == 1:
            return {"tool_calls": [{"id": "save", "name": "save_text", "arguments": {"text": "ok"}}]}
        return {"content": "done"}


def test_loop_hits_circuit_breaker_at_max_turns():
    agent = Agent(provider=EndlessProvider(), tools=[echo], max_turns=2)

    with pytest.raises(AgentLoopError, match="max_turns=2"):
        agent.run("go")


def test_unknown_tool_name_raises():
    agent = Agent(provider=UnknownProvider(), tools=[echo])

    with pytest.raises(ToolNotFoundError):
        agent.run("go")


def test_consequential_tool_confirm_callback_called_before_execute():
    calls = []

    def confirm(tool, arguments):
        calls.append((tool.name, arguments))
        return True

    agent = Agent(provider=ConsequentialProvider(), tools=[save_text], confirm_callback=confirm)
    result = agent.run("go")

    assert result.output == "done"
    assert calls == [("save_text", {"text": "ok"})]
    assert result.tool_call_count == 1
