from __future__ import annotations

from dataclasses import dataclass

from deepagents.backends.protocol import (
    EditResult,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
    GrepMatch,
    WriteResult,
)
from deepagents.backends.repl import MontyREPL


@dataclass
class _SpyBackend:
    ls_info_calls: list[str]

    def ls_info(self, path: str) -> list[FileInfo]:
        self.ls_info_calls.append(path)
        return [{"path": "/x"}]

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return f"read:{file_path}:{offset}:{limit}"

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(path=file_path)

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ) -> EditResult:
        return EditResult(path=file_path, occurrences=1)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return [{"path": "/g"}]

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch] | str:
        return [{"path": "/f", "line": 1, "text": "t"}]

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        return [FileUploadResponse(path=p) for p, _ in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return [FileDownloadResponse(path=p, content=b"x") for p in paths]


def test_monty_repl_proxies_backend_methods() -> None:
    backend = _SpyBackend(ls_info_calls=[])
    repl = MontyREPL(backend=backend)  # type: ignore[arg-type]

    assert repl.ls_info("/a") == [{"path": "/x"}]
    assert backend.ls_info_calls == ["/a"]
    assert repl.read("/f", offset=1, limit=2) == "read:/f:1:2"
    assert repl.write("/w", "c") == WriteResult(path="/w")
    assert repl.edit("/e", "o", "n") == EditResult(path="/e", occurrences=1)
    assert repl.glob_info("*.py") == [{"path": "/g"}]
    assert repl.grep_raw("p") == [{"path": "/f", "line": 1, "text": "t"}]
    assert repl.upload_files([("/u", b"y")]) == [FileUploadResponse(path="/u")]
    assert repl.download_files(["/d"]) == [FileDownloadResponse(path="/d", content=b"x")]


def test_monty_repl_has_id() -> None:
    backend = _SpyBackend(ls_info_calls=[])
    repl = MontyREPL(backend=backend)  # type: ignore[arg-type]
    assert isinstance(repl.id, str)
    assert repl.id


def test_monty_repl_repl_runs_code() -> None:
    backend = _SpyBackend(ls_info_calls=[])
    repl = MontyREPL(backend=backend)  # type: ignore[arg-type]

    result = repl.repl("1 + 1")
    assert result.error is None
    assert result.output == "2"
