from findex.models import DocMeta, Index, Posting
from findex.query import And, Not, Or, Phrase, Term, parse


def test_or_binds_looser_than_and():
    assert parse("a OR b c") == Or(Term("a"), And(Term("b"), Term("c")))


def test_parens_not_and_phrase():
    got = parse('python AND (async OR await) NOT java "event loop"')
    expect = And(
        And(
            And(Term("python"), Or(Term("async"), Term("await"))),
            Not(Term("java")),
        ),
        Phrase(("event", "loop")),
    )
    assert got == expect


def test_operators_build_the_same_tree():
    built = Term("python") & (Term("async") | Term("await")) & ~Term("java")
    assert built == And(
        And(Term("python"), Or(Term("async"), Term("await"))),
        Not(Term("java")),
    )


def test_phrase_needs_adjacent_positions():
    postings = {
        "event": [Posting(0, 1, (3,)), Posting(1, 1, (1,))],
        "loop": [Posting(0, 1, (4,)), Posting(1, 1, (8,))],
    }
    meta = {i: DocMeta(str(i), str(i)) for i in range(2)}
    index = Index(postings, {0: 5, 1: 9}, meta)
    assert Phrase(("event", "loop")).evaluate(index) == {0}
