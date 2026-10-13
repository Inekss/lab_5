from __future__ import annotations

import functools
import heapq
import logging
import re
from pathlib import Path
from time import perf_counter

from findex.deco import log, timed
from findex.documents import read_text
from findex.index import OUT_PKL
from findex.models import Index, SearchResult
from findex.query import parse, query_terms
from findex.score import BM25, TfIdf
from findex.store import open_index

QUERY = '(quirrell OR buckbeak) AND harry NOT lockhart "privet drive"'
K = 5
# hand labels: the book that query is about, must show up in the top 5
EVAL = (
    ("quirrell", "harry_potter_1"),
    ("flamel", "harry_potter_1"),
    ("firenze", "harry_potter_1"),
    ("basilisk", "harry_potter_2"),
    ("lockhart", "harry_potter_2"),
    ("dobby", "harry_potter_2"),
    ('"chamber of secrets"', "harry_potter_2"),
    ("buckbeak", "harry_potter_3"),
    ("lupin", "harry_potter_3"),
    ("pettigrew", "harry_potter_3"),
)

ENGINE = "merge"  # merge | set
BENCH_REPEATS = 200000


def merge_and(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def merge_or(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def merge_not(a: list[int], b: list[int]) -> list[int]:
    i = j = 0
    out: list[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            j += 1
    out.extend(a[i:])
    return out


def set_and(a: list[int], b: list[int]) -> list[int]:
    return sorted(set(a) & set(b))


def set_or(a: list[int], b: list[int]) -> list[int]:
    return sorted(set(a) | set(b))


def set_not(a: list[int], b: list[int]) -> list[int]:
    return sorted(set(a) - set(b))


_OPS = {
    "merge": {"AND": merge_and, "OR": merge_or, "NOT": merge_not},
    "set": {"AND": set_and, "OR": set_or, "NOT": set_not},
}


def _ids(index: Index, term: str) -> list[int]:
    return [p.doc_id for p in index.postings.get(term, ())]


def boolean_search(index: Index, query: str, engine: str) -> list[int]:
    # "term term" is AND; OR / NOT bind the next term; left to right
    universe = sorted(index.doc_meta)
    acc: list[int] | None = None
    op = "AND"
    fn = _OPS[engine]
    for raw in query.split():
        word = raw.upper()
        if word in ("AND", "OR", "NOT"):
            op = word
            continue
        docs = _ids(index, raw.lower())
        if acc is None:
            acc = fn["NOT"](universe, docs) if op == "NOT" else docs
        else:
            acc = fn[op](acc, docs)
        op = "AND"
    return [] if acc is None else acc


def _freq(index: Index, term: str) -> int:
    return sum(p.tf for p in index.postings[term])


def bench(index: Index) -> None:
    if len(index.postings) < 2:
        print("bench skipped")
        return
    ranked = sorted(index.postings, key=lambda t: (_freq(index, t), t))
    pairs = (("rare", ranked[:2]), ("common", ranked[-2:]))
    for label, (left, right) in pairs:
        query = f"{left} {right}"
        for engine in ("merge", "set"):
            t0 = perf_counter()
            for _ in range(BENCH_REPEATS):
                boolean_search(index, query, engine)
            avg = (perf_counter() - t0) / BENCH_REPEATS
            print(
                f"{label} {left!r} tf={_freq(index, left)} "
                f"{right!r} tf={_freq(index, right)} {engine} {avg:.8f}s"
            )


def _matcher(index: Index):
    fn = getattr(index, "_match_ids", None)
    if fn is not None:
        return fn

    @functools.lru_cache(maxsize=256)
    def match_ids(query: str) -> tuple[int, ...]:
        log.debug("cache miss %s", query)
        return tuple(sorted(parse(query).evaluate(index)))

    index._match_ids = match_ids
    return match_ids


def _snippet(index: Index, doc_id: int, terms: list[str]) -> str:
    path = Path(index.doc_meta[doc_id].path)
    if not path.is_file():
        return ""
    text = read_text(path)
    folded = text.lower()
    at = -1
    hit = ""
    for term in sorted(set(terms), key=lambda t: (index.df(t) or 10**9, t)):
        found = folded.find(term)
        if found >= 0:
            at, hit = found, term
            break
    if at < 0:
        window = text[:160]
    else:
        window = text[max(0, at - 80) : at + len(hit) + 80]
    for term in sorted(set(terms), key=len, reverse=True):
        if not term:
            continue
        window = re.sub(
            rf"(?i)(?<!\w){re.escape(term)}(?!\w)",
            lambda m: f"[{m.group(0)}]",
            window,
        )
    return window.replace("\n", " ")


@timed
def search(index: Index, query: str, scorer, k: int = 10) -> list[SearchResult]:
    fn = _matcher(index)
    before = fn.cache_info().hits
    ids = set(fn(query))
    if fn.cache_info().hits > before:
        log.debug("cache hit %s", query)
    node = parse(query)
    terms = query_terms(node)
    scores: dict[int, float] = {}
    for term in terms:
        if term not in index:
            continue
        for posting in index[term]:
            if posting.doc_id not in ids:
                continue
            scores[posting.doc_id] = scores.get(posting.doc_id, 0.0) + scorer(term, posting, index)
    top = heapq.nlargest(k, scores.items(), key=lambda item: item[1])
    return [
        SearchResult(score, doc_id, index.doc_meta[doc_id].title, _snippet(index, doc_id, terms))
        for doc_id, score in top
    ]


def precision_at_5(index: Index) -> None:
    print(f"{'query':<28} {'tfidf':>8} {'bm25':>8}")
    totals = [0.0, 0.0]
    for query, title in EVAL:
        row = []
        for scorer in (TfIdf(), BM25()):
            hits = search(index, query, scorer, K)
            row.append(sum(h.title == title for h in hits) / K)
        totals[0] += row[0]
        totals[1] += row[1]
        print(f"{query:<28} {row[0]:8.2f} {row[1]:8.2f}")
    n = len(EVAL)
    print(f"{'mean':<28} {totals[0] / n:8.2f} {totals[1] / n:8.2f}")


def _show(index: Index, query: str, scorer) -> None:
    print(f"\n{query} ({type(scorer).__name__})")
    for rank, hit in enumerate(search(index, query, scorer, K), 1):
        print(f"{rank} {hit.score:.4f} {hit.title} {hit.snippet}")


def main() -> None:
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    with open_index(OUT_PKL) as index:
        _show(index, QUERY, BM25())
        _show(index, QUERY, BM25())
        print("cache", index._match_ids.cache_info())


if __name__ == "__main__":
    main()
