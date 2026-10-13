from __future__ import annotations

import math
from typing import Literal, Protocol

from findex.models import Index, Posting

type ScorerName = Literal["bm25", "tfidf"]


class Scorer(Protocol):
    def __call__(self, term: str, posting: Posting, index: Index) -> float: ...


class TfIdf:
    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        idf = math.log(index.num_docs / index.df(term))
        return (1 + math.log(posting.tf)) * idf


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b

    def __call__(self, term: str, posting: Posting, index: Index) -> float:
        idf = math.log(index.num_docs / index.df(term))
        dl = index.doc_length(posting.doc_id)
        norm = 1 - self.b + self.b * dl / index.avg_doc_length
        tf = posting.tf
        return idf * (tf * (self.k1 + 1)) / (tf + self.k1 * norm)


def get_scorer(name: ScorerName) -> Scorer:
    if name == "tfidf":
        return TfIdf()
    return BM25()


def sanity(index: Index) -> None:
    # same tf and doc length; only df changes, so the rare term weighs more
    rare, common = "quirrell", "norbert"
    bm = BM25()
    print(
        f"rare {rare} df={index.df(rare)} w={bm(rare, Posting(0, 1), index):.4f}  "
        f"common {common} df={index.df(common)} w={bm(common, Posting(0, 1), index):.4f}"
    )
    # 20th hit adds almost nothing next to the first
    for tf in (1, 2, 20, 21):
        w = bm(rare, Posting(0, tf), index)
        before = bm(rare, Posting(0, tf - 1), index) if tf > 1 else 0.0
        print(f"tf={tf} w={w:.4f} step={w - before:.4f}")
    short = min(index.doc_lengths, key=lambda doc_id: index.doc_lengths[doc_id])
    long = max(index.doc_lengths, key=lambda doc_id: index.doc_lengths[doc_id])
    flat = BM25(b=0)
    for doc_id in (short, long):
        one = Posting(doc_id, 1)
        print(
            f"len={index.doc_length(doc_id)} {index.doc_meta[doc_id].title} "
            f"b0.75={bm(rare, one, index):.4f} b0={flat(rare, one, index):.4f}"
        )
