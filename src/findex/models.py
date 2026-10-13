from __future__ import annotations

import functools
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field


@dataclass
class StatsResult:
    n_docs: int
    n_tokens: int
    vocab_size: int
    top50: list[tuple[str, int]]
    elapsed_s: float
    peak_kib: float


@dataclass(frozen=True)
class PostingPlain:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int
    positions: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DocMeta:
    path: str
    title: str


@dataclass(frozen=True, slots=True)
class PartialIndex:
    postings: dict[str, list[Posting]]
    doc_lengths: dict[int, int]
    doc_meta: dict[int, DocMeta]


@dataclass(order=True, frozen=True, slots=True)
class SearchResult:
    score: float
    doc_id: int
    title: str = field(compare=False)
    snippet: str = field(compare=False)


class Index(Mapping[str, list[Posting]]):
    # not frozen: cached_property writes avg_doc_length onto the instance
    def __init__(
        self,
        postings: dict[str, list[Posting]],
        doc_lengths: dict[int, int],
        doc_meta: dict[int, DocMeta],
    ) -> None:
        self.postings = postings
        self.doc_lengths = doc_lengths
        self.doc_meta = doc_meta
        self._match_ids: Callable[[str], tuple[int, ...]] | None = None

    def __getitem__(self, term: str) -> list[Posting]:
        return self.postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __len__(self) -> int:
        return len(self.postings)

    def __repr__(self) -> str:
        return f"Index(terms={len(self):_}, docs={self.num_docs:_})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Index):
            return False
        return (
            self.postings == other.postings
            and self.doc_lengths == other.doc_lengths
            and self.doc_meta == other.doc_meta
        )

    @property
    def num_docs(self) -> int:
        return len(self.doc_lengths)

    @functools.cached_property
    def avg_doc_length(self) -> float:
        return sum(self.doc_lengths.values()) / self.num_docs

    def doc_length(self, doc_id: int) -> int:
        return self.doc_lengths[doc_id]

    def df(self, term: str) -> int:
        return len(self[term]) if term in self else 0
