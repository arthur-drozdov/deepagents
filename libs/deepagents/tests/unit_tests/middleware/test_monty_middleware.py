from __future__ import annotations

from langchain.tools import ToolRuntime
from langgraph.types import Command

from deepagents.backends.state import StateBackend
from deepagents.middleware.repl import MontyMiddleware


class _DummyRuntimeState(dict):
    pass


def test_monty_middleware_repl_tool_returns_command_and_updates_state() -> None:
    mw = MontyMiddleware(backend=StateBackend)
    tool = next(t for t in mw.tools if t.name == "repl")

    state = _DummyRuntimeState()
    runtime = ToolRuntime(state=state, context=None, config={}, stream_writer=None, tool_call_id="x", store=None)  # type: ignore[arg-type]

    result = tool.func(code="1 + 1", runtime=runtime, timeout=5)

    assert isinstance(result, Command)
    assert isinstance(result.update["repl_state"], (str, bytes))
    assert result.update["messages"][0].content == "2"


def test_monty_middleware_sets_repl_state_key() -> None:
    mw = MontyMiddleware(backend=StateBackend)
    tool = next(t for t in mw.tools if t.name == "repl")

    state = _DummyRuntimeState()
    runtime = ToolRuntime(state=state, context=None, config={}, stream_writer=None, tool_call_id="x", store=None)  # type: ignore[arg-type]

    result = tool.func(code="1 + 1", runtime=runtime)

    assert "repl_state" in result.update
    assert isinstance(result.update["repl_state"], (str, bytes))
