from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_core.messages import HumanMessage

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.repl import MontyREPL
from deepagents.graph import create_deep_agent


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        backend = MontyREPL(backend=FilesystemBackend(root_dir=root, virtual_mode=True))
        agent = create_deep_agent(backend=backend)

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=("Use the repl to compute 523! and Use the repl to write the result to using pathlib /out.txt, then respond done.")
                    )
                ]
            }
        )

        print(result["messages"][-1].content)
        print((root / "out.txt").read_text())


if __name__ == "__main__":
    main()
