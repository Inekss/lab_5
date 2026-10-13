import pytest

from findex.documents import CorpusReader
from findex.index import build, build_all, build_partial, jobs, merge
from findex.tokenizer import Tokenizer


def _write(folder, names: list[str]) -> None:
    for name in names:
        (folder / name).write_text(f"{name} alpha alpha\n", encoding="utf-8")


def test_serial_partial_matches_build(tmp_path):
    _write(tmp_path, ["a.txt", "b.txt"])
    paths = sorted(tmp_path.glob("*.txt"))
    got = merge([build_partial((paths, 0, True))])
    assert got == build(CorpusReader(tmp_path), Tokenizer(), True)


def test_executors_match(tmp_path):
    _write(tmp_path, ["a.txt", "b.txt", "c.txt"])
    paths = sorted(tmp_path.glob("*.txt"))
    indexes = [build_all(paths, 2, ex, True)[0] for ex in ("serial", "threads", "processes")]
    assert indexes[0] == indexes[1] == indexes[2]


def test_doc_ids_are_the_path_order(tmp_path):
    _write(tmp_path, ["b.txt", "a.txt", "c.txt"])
    paths = sorted(tmp_path.glob("*.txt"))
    index = merge([build_partial(job) for job in jobs(paths, 2, True)])
    assert sorted(index.doc_lengths) == list(range(len(paths)))
    for plist in index.postings.values():
        ids = [posting.doc_id for posting in plist]
        assert ids == sorted(ids)


def test_worker_error_surfaces(tmp_path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(FileNotFoundError):
        build_all([missing], 1, "processes", False)
