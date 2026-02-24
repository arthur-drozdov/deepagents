"""End-to-end unit test for deepagents using Monty-backed repl tool."""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.repl import MontyREPL
from deepagents.graph import create_deep_agent
from tests.unit_tests.test_end_to_end import FixedGenericFakeChatModel


def test_deep_agent_with_fake_llm_uses_monty_repl_to_write_file(tmp_path: Path) -> None:
    out_virtual_path = "/out.txt"
    out_path = tmp_path / "out.txt"

    model = FixedGenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "repl",
                            "args": {
                                "code": (f"from pathlib import Path\nresult = 2 + 2\nPath({out_virtual_path!r}).write_text(str(result))\nresult"),
                                "timeout": 5,
                            },
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )

    backend = MontyREPL(backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True))
    agent = create_deep_agent(model=model, backend=backend)

    result = agent.invoke({"messages": [HumanMessage(content="Compute 2+2 and write it to out.txt")]})

    assert "messages" in result
    assert out_path.read_text() == "4"
