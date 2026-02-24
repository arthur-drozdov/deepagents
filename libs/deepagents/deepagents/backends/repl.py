"""Monty-backed REPL backend.

This backend wraps an existing `BackendProtocol` to provide REPL-style evaluation
via Monty, and exposes filesystem-like access to the wrapped backend through
Monty's `AbstractOS` interface.
"""

from __future__ import annotations

import uuid
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

import pydantic_monty
from pydantic_monty import AbstractOS, ResourceLimits, StatResult

from deepagents.backends.protocol import (
    BackendProtocol,
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    ReplBackendProtocol,
    ReplResponse,
    WriteResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable


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


class MontyREPL(ReplBackendProtocol):
    """REPL backend that evaluates code using Monty."""

    def __init__(
        self,
        backend: BackendProtocol,
        *,
        script_name: str = "repl.py",
        inputs: list[str] | None = None,
        external_functions: list[str] | None = None,
        external_function_implementations: dict[str, Callable[..., Any]] | None = None,
        type_check: bool = False,
        type_check_stubs: str | None = None,
    ) -> None:
        """Create a Monty-backed REPL.

        Args:
            backend: Backend used for file operations exposed to Monty via `pathlib`.
            script_name: Script name reported to Monty.
            inputs: Monty inputs list.
            external_functions: Names of external functions Monty may call.
            external_function_implementations: Host implementations for external functions.
            type_check: Whether to type-check the provided code.
            type_check_stubs: Optional stubs used for type-checking.
        """
        self._id = str(uuid.uuid4())
        self._backend = backend
        self._script_name = script_name
        self._inputs = inputs
        self._external_functions = external_functions
        self._type_check = type_check
        self._type_check_stubs = type_check_stubs
        self._external_function_implementations = external_function_implementations

    @property
    def id(self) -> str:
        """Unique identifier for this REPL backend instance."""
        return self._id

    def repl(self, code: str, *, timeout: int | None = None) -> ReplResponse:
        """Evaluate code with Monty.

        Args:
            code: Code string to evaluate.
            timeout: Optional maximum duration in seconds.

        Returns:
            ReplResponse containing output and optional error.
        """
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
            return ReplResponse(output="", error=str(e))

        limits = ResourceLimits()
        if timeout is not None:
            limits["max_duration_secs"] = timeout

        try:
            result = m.run(
                os=_MontyOS(self._backend),
                limits=limits,
                external_functions=self._external_function_implementations,
            )
        except Exception as e:  # noqa: BLE001
            return ReplResponse(output="", error=str(e))

        return ReplResponse(output=str(result))

    def ls_info(self, path: str) -> list[FileInfo]:
        """Proxy to the wrapped backend."""
        return self._backend.ls_info(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Proxy to the wrapped backend."""
        return self._backend.read(file_path, offset=offset, limit=limit)

    def write(self, file_path: str, content: str) -> WriteResult:
        """Proxy to the wrapped backend."""
        return self._backend.write(file_path, content)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        """Proxy to the wrapped backend."""
        return self._backend.edit(file_path, old_string, new_string, replace_all=replace_all)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Proxy to the wrapped backend."""
        return self._backend.glob_info(pattern, path=path)

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch] | str:
        """Proxy to the wrapped backend."""
        return self._backend.grep_raw(pattern, path=path, glob=glob)

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Proxy to the wrapped backend."""
        return self._backend.upload_files(files)

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Proxy to the wrapped backend."""
        return self._backend.download_files(paths)
