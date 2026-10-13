from __future__ import annotations

import gc
import multiprocessing as mp
import os
import tracemalloc
from array import array
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from time import perf_counter, process_time

from findex.deco import timed
from findex.documents import CorpusReader, read_text
from findex.models import DocMeta, Index, PartialIndex, Posting, PostingPlain
from findex.store import pickle_size_and_load, timed_save_load
from findex.tokenizer import Tokenizer

CORPUS = Path(__file__).resolve().parents[1] / "resources"
OUT_PKL = Path(__file__).resolve().parents[2] / "index.pkl"
OUT_JSON = Path(__file__).resolve().parents[2] / "index.json"
WITH_POSITIONS = True  # phrase queries need token offsets
WORKERS = os.cpu_count() or 1
EXECUTOR = "processes"  # "serial" | "threads" | "processes"

Job = tuple[list[Path], int, bool]


def build_partial(job: Job) -> PartialIndex:
    # module-level so spawn can import it; a lambda cannot cross the process boundary
    # worker reads the path; the parent sent the path and the start doc_id, not the text
    paths, start_id, with_positions = job
    tok = Tokenizer()
    postings: dict[str, list[Posting]] = defaultdict(list)
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    for offset, path in enumerate(paths):
        doc_id = start_id + offset
        text = read_text(path)
        counts: Counter[str] = Counter()
        pos: dict[str, list[int]] | None = defaultdict(list) if with_positions else None
        n = 0
        for i, term in enumerate(tok.tokenize(text)):
            counts[term] += 1
            n += 1
            if pos is not None:
                pos[term].append(i)
        doc_lengths[doc_id] = n
        doc_meta[doc_id] = DocMeta(str(path), path.stem)
        for term, tf in counts.items():
            positions = tuple(pos[term]) if pos is not None else ()
            postings[term].append(Posting(doc_id, tf, positions))
    return PartialIndex(dict(postings), doc_lengths, doc_meta)


def jobs(paths: list[Path], workers: int, with_positions: bool) -> list[Job]:
    n = max(1, workers)
    size = len(paths)
    out: list[Job] = []
    for i in range(n):
        a = i * size // n
        b = (i + 1) * size // n
        # a is the first doc_id (index in the full sorted list). Empty slices still run.
        out.append((paths[a:b], a, with_positions))
    return out


def merge(parts: list[PartialIndex]) -> Index:
    postings: dict[str, list[Posting]] = defaultdict(list)
    lengths: dict[int, int] = {}
    meta: dict[int, DocMeta] = {}
    for part in parts:
        for term, plist in part.postings.items():
            postings[term].extend(plist)
        lengths.update(part.doc_lengths)
        meta.update(part.doc_meta)
    # sort so chunk finish order does not change the index
    for plist in postings.values():
        plist.sort(key=lambda p: p.doc_id)
    return Index(dict(postings), lengths, meta)


def _timed_partial(job: Job) -> tuple[PartialIndex, float]:
    c0 = process_time()
    return build_partial(job), process_time() - c0


def build_all(
    paths: list[Path],
    workers: int,
    executor: str,
    with_positions: bool,
) -> tuple[Index, float, float]:
    n = 1 if executor == "serial" else workers
    work = jobs(paths, n, with_positions)
    if executor == "serial":
        timed_parts = [_timed_partial(job) for job in work]
    elif executor == "threads":
        with ThreadPoolExecutor(n) as pool:
            timed_parts = list(pool.map(_timed_partial, work))
    else:
        # spawn: the child imports this module. map re-raises the worker traceback.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(n, mp_context=ctx) as pool:
            timed_parts = list(pool.map(_timed_partial, work))
    parts = [part for part, _ in timed_parts]
    worker_cpu = sum(cpu for _, cpu in timed_parts)
    t0 = perf_counter()
    index = merge(parts)
    return index, perf_counter() - t0, worker_cpu


@timed
def build(reader: CorpusReader, tok: Tokenizer, with_positions: bool = False) -> Index:
    paths = sorted(reader.path.glob("*.txt"))
    # tok stays on the signature; the worker builds its own Tokenizer
    return merge([build_partial((paths, 0, with_positions))])


def build_variant(reader: CorpusReader, tok: Tokenizer, kind: str):
    if kind == "array":
        # 'I' = 4-byte unsigned ints, no Posting objects
        postings = defaultdict(lambda: (array("I"), array("I")))
    else:
        postings = defaultdict(list)
    cls = PostingPlain if kind == "plain" else Posting
    doc_lengths: dict[int, int] = {}
    doc_meta: dict[int, DocMeta] = {}
    for doc_id, path, text in reader.iter_docs():
        counts = Counter(tok.tokenize(text))
        doc_lengths[doc_id] = sum(counts.values())
        doc_meta[doc_id] = DocMeta(str(path), path.stem)
        for term, tf in counts.items():
            if kind == "array":
                postings[term][0].append(doc_id)
                postings[term][1].append(tf)
            else:
                postings[term].append(cls(doc_id, tf))
    return {"postings": dict(postings), "doc_lengths": doc_lengths, "doc_meta": doc_meta}


def _peak_kib(fn):
    gc.collect()
    tracemalloc.start()
    obj = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return obj, peak / 1024


def positions_cost() -> None:
    reader, tok = CorpusReader(CORPUS), Tokenizer()
    for flag in (False, True):
        index, peak = _peak_kib(lambda flag=flag: build(reader, tok, flag))
        size, load_s = pickle_size_and_load(index)
        print(f"positions={flag} peak={peak:.1f}KiB pickle={size} load={load_s:.4f}s")
        del index
        gc.collect()


def memory_study() -> None:
    reader, tok = CorpusReader(CORPUS), Tokenizer()
    for kind in ("plain", "slots", "array"):
        obj, peak = _peak_kib(lambda kind=kind: build_variant(reader, tok, kind))
        size, load_s = pickle_size_and_load(obj)
        print(f"{kind} peak={peak:.1f}KiB file={size} load={load_s:.4f}s")
        del obj
        gc.collect()


def main() -> Index:
    paths = sorted(CORPUS.glob("*.txt"))
    tracemalloc.start()
    t0 = perf_counter()
    index, merge_s, _worker_cpu = build_all(paths, WORKERS, EXECUTOR, WITH_POSITIONS)
    elapsed = perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(
        f"{index!r} tokens={sum(index.doc_lengths.values())} "
        f"time={elapsed:.4f}s peak={peak / 1024:.1f}KiB "
        f"executor={EXECUTOR} workers={WORKERS} merge={merge_s:.4f}s"
    )
    timed_save_load(index, OUT_PKL, OUT_JSON)
    return index


if __name__ == "__main__":
    main()
