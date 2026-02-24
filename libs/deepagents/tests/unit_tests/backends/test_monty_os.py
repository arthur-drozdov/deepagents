from __future__ import annotations

from deepagents.backends.protocol import (
    BackendProtocol,
    FileDownloadResponse,
    FileInfo,
    FileUploadResponse,
)
from deepagents.backends.repl import _MontyOS


class _Backend(BackendProtocol):
    def __init__(self) -> None:
        self._files = {"/a.txt": b"hi"}
        self._dirs = {"/"}

    def ls_info(self, path: str) -> list[FileInfo]:
        if path == "/":
            return [{"path": "/a.txt", "is_dir": False, "size": 2}]
        return []

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        raise NotImplementedError

    def write(self, file_path: str, content: str):
        self._files[file_path] = content.encode("utf-8")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,  # noqa: FBT001, FBT002
    ):
        raise NotImplementedError

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        raise NotImplementedError

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None):
        raise NotImplementedError

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        for p, b in files:
            self._files[p] = b
        return [FileUploadResponse(path=p) for p, _ in files]

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        out: list[FileDownloadResponse] = []
        for p in paths:
            if p not in self._files:
                out.append(FileDownloadResponse(path=p, content=None, error="file_not_found"))
            else:
                out.append(FileDownloadResponse(path=p, content=self._files[p], error=None))
        return out


def test_monty_os_read_and_exists() -> None:
    os = _MontyOS(_Backend())
    pp = __import__("pathlib").PurePosixPath
    assert os.path_exists(pp("/a.txt")) is True
    assert os.path_is_file(pp("/a.txt")) is True
    assert os.path_read_text(pp("/a.txt")) == "hi"


def test_monty_os_iterdir_root() -> None:
    os = _MontyOS(_Backend())
    pp = __import__("pathlib").PurePosixPath
    entries = os.path_iterdir(pp("/"))
    assert entries == [pp("/a.txt")]
