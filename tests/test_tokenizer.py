from findex.tokenizer import Tokenizer


def test_nfkc_and_lower():
    assert list(Tokenizer().tokenize("CAFÉ")) == ["café"]
    assert list(Tokenizer().tokenize("Ｈａｒｒｙ")) == ["harry"]


def test_apostrophe_kept():
    assert list(Tokenizer().tokenize("don't")) == ["don't"]
    assert list(Tokenizer().tokenize("it's")) == ["it's"]


def test_hyphenated_word():
    assert list(Tokenizer().tokenize("good-for-nothing")) == ["good-for-nothing"]


def test_digits_kept():
    assert list(Tokenizer().tokenize("number 4 and 1991")) == ["number", "4", "and", "1991"]


def test_punctuation_splits():
    assert list(Tokenizer().tokenize("Hello, world!")) == ["hello", "world"]


def test_empty_and_punct_only():
    assert list(Tokenizer().tokenize("")) == []
    assert list(Tokenizer().tokenize("... --- !!!")) == []


def test_streaming_iterator():
    gen = Tokenizer().tokenize("a b c")
    assert next(gen) == "a"
    assert list(gen) == ["b", "c"]
