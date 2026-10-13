from findex.models import DocMeta, Index, Posting
from findex.search import boolean_search as search
from findex.store import load_json, load_pickle, save_json, save_pickle


def _index() -> Index:
    postings = {
        "a": [Posting(0, 1), Posting(1, 1), Posting(2, 1)],
        "b": [Posting(1, 1), Posting(2, 1)],
        "c": [Posting(2, 1)],
    }
    meta = {i: DocMeta(str(i), str(i)) for i in range(4)}
    lengths = {i: 1 for i in range(4)}
    return Index(postings, lengths, meta)


def test_frozen_records_go_in_a_set():
    found = {Posting(1, 2), DocMeta("a", "b")}
    assert Posting(1, 2) in found
    assert DocMeta("a", "b") in found


def test_merge_and_set_agree():
    index = _index()
    queries = ("a b", "a OR c", "a NOT b", "NOT a")
    expect = ([1, 2], [0, 1, 2], [0], [3])
    for query, hits in zip(queries, expect):
        assert search(index, query, "merge") == hits
        assert search(index, query, "set") == hits


def test_pickle_and_json_roundtrip(tmp_path):
    index = _index()
    pkl, js = tmp_path / "i.pkl", tmp_path / "i.json"
    save_pickle(index, pkl)
    save_json(index, js)
    assert load_pickle(pkl) == index
    assert load_json(js) == index
