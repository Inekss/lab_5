from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path


def read_text(path: Path) -> str:
    # utf-8 first; a few books are cp1251 and fail on the first byte
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1251")


class CorpusReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def iter_documents(self) -> Iterator[str]:
        for file in self.path.glob("*.txt"):
            yield read_text(file)

    def iter_docs(self) -> Iterator[tuple[int, Path, str]]:
        # sorted paths keep doc_id stable; ids only increase, so postings stay in order
        for doc_id, file in enumerate(sorted(self.path.glob("*.txt"))):
            yield doc_id, file, read_text(file)
