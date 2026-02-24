"""Middleware for providing a Monty-backed repl tool to an agent."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Annotated, Any, NotRequired

import pydantic_monty
from pydantic_monty import AbstractOS, ResourceLimits, StatResult

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ContextT, ResponseT
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command

from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

if TYPE_CHECKING:
    from collections.abc import Callable


class MontyState(AgentState):
    repl_state: NotRequired[str]


class _MontyOS(AbstractOS):
    def __init__(self, backend: BackendProtocol) -> None:
        self._backend = backend

    def path_exists(self, path: PurePosixPath) -> bool:
        p = str(path)
        if p == "/":
            return True
        infos = self._backend.ls_info(p)
        if infos:
            return True
        parent = str(path.parent)
        if parent == p:
            return False
        parent_infos = self._backend.ls_info(parent)
        return any(info.get("path") == p for info in parent_infos)

    def path_is_file(self, path: PurePosixPath) -> bool:
        p = str(path)
        if p == "/":
            return False
        parent_infos = self._backend.ls_info(str(path.parent))
        for info in parent_infos:
            if info.get("path") == p:
                return not bool(info.get("is_dir", False))
        res = self._backend.download_files([p])[0]
        return res.error is None and res.content is not None

    def path_is_dir(self, path: PurePosixPath) -> bool:
        p = str(path)
        if p == "/":
            return True
        parent_infos = self._backend.ls_info(str(path.parent))
        for info in parent_infos:
            if info.get("path") == p:
                return bool(info.get("is_dir", False))
        return bool(self._backend.ls_info(p))

    def path_is_symlink(self, path: PurePosixPath) -> bool:  # noqa: ARG002
        return False

    def path_read_text(self, path: PurePosixPath) -> str:
        p = str(path)
        res = self._backend.download_files([p])[0]
        if res.error is not None or res.content is None:
            raise FileNotFoundError(p)
        return res.content.decode("utf-8")

    def path_read_bytes(self, path: PurePosixPath) -> bytes:
        p = str(path)
        res = self._backend.download_files([p])[0]
        if res.error is not None or res.content is None:
            raise FileNotFoundError(p)
        return res.content

    def path_write_text(self, path: PurePosixPath, data: str) -> None:
        self._backend.write(str(path), data)

    def path_write_bytes(self, path: PurePosixPath, data: bytes) -> None:
        self._backend.upload_files([(str(path), data)])

    def path_mkdir(
        self,
        path: PurePosixPath,  # noqa: ARG002
        parents: bool,  # noqa: FBT001, ARG002
        exist_ok: bool,  # noqa: FBT001
    ) -> None:
        if not exist_ok:
            msg = "mkdir with exist_ok=False is not supported"
            raise NotImplementedError(msg)

    def path_unlink(self, path: PurePosixPath) -> None:
        msg = "unlink is not supported"
        raise NotImplementedError(msg)

    def path_rmdir(self, path: PurePosixPath) -> None:
        msg = "rmdir is not supported"
        raise NotImplementedError(msg)

    def path_iterdir(self, path: PurePosixPath) -> list[PurePosixPath]:
        infos = self._backend.ls_info(str(path))
        return [PurePosixPath(info["path"]) for info in infos]

    def path_stat(self, path: PurePosixPath) -> StatResult:
        p = str(path)
        if p == "/":
            return StatResult.dir_stat(0o755, 0.0)
        parent_infos = self._backend.ls_info(str(path.parent))
        for info in parent_infos:
            if info.get("path") == p:
                if info.get("is_dir", False):
                    return StatResult.dir_stat(0o755, 0.0)
                size = int(info.get("size", 0))
                return StatResult.file_stat(size, 0o644, 0.0)
        res = self._backend.download_files([p])[0]
        if res.error is not None or res.content is None:
            raise FileNotFoundError(p)
        return StatResult.file_stat(len(res.content), 0o644, 0.0)

    def path_rename(self, path: PurePosixPath, target: PurePosixPath) -> None:
        msg = "rename is not supported"
        raise NotImplementedError(msg)

    def path_resolve(self, path: PurePosixPath) -> str:
        return str(path)

    def path_absolute(self, path: PurePosixPath) -> str:
        p = str(path)
        if p.startswith("/"):
            return p
        return "/" + p

    def getenv(self, key: str, default: str | None = None) -> str | None:  # noqa: ARG002
        return default

    def get_environ(self) -> dict[str, str]:
        return {}


class MontyMiddleware(AgentMiddleware[MontyState, ContextT, ResponseT]):
    state_schema = MontyState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES,
        script_name: str = "repl.py",
        inputs: list[str] | None = None,
        external_functions: list[str] | None = None,
        external_function_implementations: dict[str, Callable[..., Any]] | None = None,
        type_check: bool = False,
        type_check_stubs: str | None = None,
    ) -> None:
        self.backend = backend
        self._script_name = script_name
        self._inputs = inputs
        self._external_functions = external_functions
        self._external_function_implementations = external_function_implementations
        self._type_check = type_check
        self._type_check_stubs = type_check_stubs

        self.tools = [self._create_repl_tool()]

    def _get_backend(self, runtime: ToolRuntime[Any, Any]) -> BackendProtocol:
        if callable(self.backend):
            return self.backend(runtime)
        return self.backend

    def _create_repl_tool(self) -> BaseTool:
        def _run_monty(
            code: str,
            *,
            timeout: int | None,
            runtime: ToolRuntime[None, MontyState],
        ) -> Command:
            if not runtime.tool_call_id:
                msg = "Tool call ID is required for repl"
                raise ValueError(msg)

            limits = ResourceLimits()
            if timeout is not None:
                if timeout <= 0:
                    return Command(
                        update={
                            "messages": [ToolMessage(f"Error: timeout must be positive, got {timeout}.", tool_call_id=runtime.tool_call_id)],
                        }
                    )
                limits["max_duration_secs"] = timeout

            resolved_backend = self._get_backend(runtime)
            repl_state = runtime.state.get("repl_state") if isinstance(runtime.state, dict) else getattr(runtime.state, "repl_state", None)

            try:
                m = pydantic_monty.Monty(
                    code,
                    inputs=self._inputs or [],
                    external_functions=self._external_functions or [],
                    script_name=self._script_name,
                    type_check=self._type_check,
                    type_check_stubs=self._type_check_stubs,
                )
            except Exception as e:  # noqa: BLE001
                return Command(
                    update={
                        "repl_state": repl_state or "",
                        "messages": [ToolMessage(str(e), tool_call_id=runtime.tool_call_id)],
                    }
                )

            if repl_state:
                try:
                    m = pydantic_monty.Monty.load(repl_state)
                except Exception:  # noqa: BLE001
                    pass

            try:
                result = m.run(
                    os=_MontyOS(resolved_backend),
                    limits=limits,
                    external_functions=self._external_function_implementations,
                )
            except Exception as e:  # noqa: BLE001
                new_state = repl_state or ""
                try:
                    new_state = m.dump()
                except Exception:  # noqa: BLE001
                    pass
                return Command(
                    update={
                        "repl_state": new_state,
                        "messages": [ToolMessage(str(e), tool_call_id=runtime.tool_call_id)],
                    }
                )

            new_state = repl_state or ""
            try:
                new_state = m.dump()
            except Exception:  # noqa: BLE001
                pass

            return Command(
                update={
                    "repl_state": new_state,
                    "messages": [ToolMessage(str(result), tool_call_id=runtime.tool_call_id)],
                }
            )

        async def _arun_monty(
            code: Annotated[str, "Code string to evaluate in Monty."],
            runtime: ToolRuntime[None, MontyState],
            timeout: Annotated[int | None, "Optional timeout in seconds for this evaluation."] = None,
        ) -> Command:
            return _run_monty(code, timeout=timeout, runtime=runtime)

        def _sync_monty(
            code: Annotated[str, "Code string to evaluate in Monty."],
            runtime: ToolRuntime[None, MontyState],
            timeout: Annotated[int | None, "Optional timeout in seconds for this evaluation."] = None,
        ) -> Command:
            return _run_monty(code, timeout=timeout, runtime=runtime)

        return StructuredTool.from_function(
            name="repl",
            description="Evaluate code using Monty. State is persisted in agent state under 'repl_state'.",
            func=_sync_monty,
            coroutine=_arun_monty,
        )
