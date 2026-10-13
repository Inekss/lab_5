from __future__ import annotations

import json
import os
import pickle
import tempfile
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from findex.deco import timed
from findex.models import DocMeta, Index, Posting


def save_pickle(obj, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


@timed
def load_pickle(path: Path):
    # only load an index this program wrote: pickle.load runs the file's reduce
    with path.open("rb") as f:
        return pickle.load(f)


@timed
def _unpickle(f):
    return pickle.load(f)


@contextmanager
def open_index(path: Path):
    f = path.open("rb")
    try:
        yield _unpickle(f)
    finally:
        f.close()


def save_json(index: Index, path: Path) -> None:
    data = {
        "doc_lengths": {str(k): v for k, v in index.doc_lengths.items()},
        "doc_meta": {
            str(k): {"path": m.path, "title": m.title} for k, m in index.doc_meta.items()
        },
        "postings": {
            term: [[p.doc_id, p.tf, list(p.positions)] for p in plist]
            for term, plist in index.postings.items()
        },
    }
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Index:
    data = json.loads(path.read_text(encoding="utf-8"))
    postings = {
        term: [Posting(p[0], p[1], tuple(p[2])) for p in plist]
        for term, plist in data["postings"].items()
    }
    # JSON object keys are strings
    doc_lengths = {int(k): v for k, v in data["doc_lengths"].items()}
    doc_meta = {int(k): DocMeta(v["path"], v["title"]) for k, v in data["doc_meta"].items()}
    return Index(postings, doc_lengths, doc_meta)


def _median(fn, repeats: int = 5) -> float:
    samples = []
    for _ in range(repeats):
        t0 = perf_counter()
        fn()
        samples.append(perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def pickle_size_and_load(obj) -> tuple[int, float]:
    fd, name = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    path = Path(name)
    try:
        save_pickle(obj, path)
        return path.stat().st_size, _median(lambda: load_pickle(path))
    finally:
        path.unlink(missing_ok=True)


def timed_save_load(index: Index, pkl: Path, js: Path) -> None:
    save_p = _median(lambda: save_pickle(index, pkl))
    load_p = _median(lambda: load_pickle(pkl))
    save_j = _median(lambda: save_json(index, js))
    load_j = _median(lambda: load_json(js))
    print(
        f"pickle size={pkl.stat().st_size} save={save_p:.4f}s load={load_p:.4f}s\n"
        f"json size={js.stat().st_size} save={save_j:.4f}s load={load_j:.4f}s"
    )
