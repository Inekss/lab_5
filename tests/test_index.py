import logging

import pytest

from findex.models import DocMeta, Index, Posting
from findex.score import BM25
from findex.search import search
from findex.store import open_index, save_pickle


def _index() -> Index:
    postings = {
        "rare": [Posting(0, 1)],
        "common": [Posting(0, 1), Posting(1, 1)],
    }
    meta = {0: DocMeta("short", "short"), 1: DocMeta("long", "long")}
    return Index(postings, {0: 10, 1: 10_000}, meta)


def test_mapping_protocol():
    index = _index()
    assert len(index) == 2
    assert "rare" in index and "missing" not in index
    assert index["rare"][0].doc_id == 0
    with pytest.raises(KeyError):
        index["missing"]
    assert set(index) == {"rare", "common"}
    assert repr(index) == "Index(terms=2, docs=2)"
    assert index.num_docs == 2
    assert index.df("common") == 2 and index.df("nope") == 0
    assert index.avg_doc_length == 5005


def test_bm25_rare_short_and_saturation():
    index = _index()
    bm = BM25()
    rare = bm("rare", Posting(0, 1), index)
    common = bm("common", Posting(0, 1), index)
    assert rare > common
    first = bm("rare", Posting(0, 1), index)
    step20 = bm("rare", Posting(0, 20), index) - bm("rare", Posting(0, 19), index)
    assert step20 < first
    short = bm("rare", Posting(0, 1), index)
    long = bm("rare", Posting(1, 1), index)
    assert short > long
    flat = BM25(b=0)
    assert flat("rare", Posting(0, 1), index) == flat("rare", Posting(1, 1), index)


def test_open_index_closes_on_error(tmp_path, monkeypatch):
    path = tmp_path / "i.pkl"
    save_pickle(_index(), path)
    closed = {}
    real_open = type(path).open

    def tracking(self, *args, **kwargs):
        handle = real_open(self, *args, **kwargs)
        real_close = handle.close

        def close():
            closed["ok"] = True
            return real_close()

        handle.close = close
        return handle

    monkeypatch.setattr(type(path), "open", tracking)
    with pytest.raises(RuntimeError):
        with open_index(path):
            raise RuntimeError("boom")
    assert closed.get("ok")


def test_cache_hit_is_logged(caplog):
    index = _index()
    with caplog.at_level(logging.DEBUG, logger="findex"):
        search(index, "rare", BM25(), k=5)
        search(index, "rare", BM25(), k=5)
    assert caplog.text.count("cache miss rare") == 1
    assert "cache hit rare" in caplog.text


def test_timed_keeps_the_name():
    assert search.__name__ == "search"
