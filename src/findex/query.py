from __future__ import annotations

from dataclasses import dataclass

from findex.models import Index
from findex.tokenizer import Tokenizer


class _Q:
    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)


@dataclass(frozen=True)
class Term(_Q):
    value: str

    def evaluate(self, index: Index) -> set[int]:
        if self.value not in index:
            return set()
        return {p.doc_id for p in index[self.value]}


@dataclass(frozen=True)
class Phrase(_Q):
    words: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        # next word must sit at position p + 1
        if any(w not in index for w in self.words):
            return set()
        by_doc = [{p.doc_id: p.positions for p in index[w]} for w in self.words]
        docs = set.intersection(*(set(col) for col in by_doc))
        return {doc_id for doc_id in docs if _chain(by_doc, doc_id)}


@dataclass(frozen=True)
class And(_Q):
    left: object
    right: object

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) & self.right.evaluate(index)


@dataclass(frozen=True)
class Or(_Q):
    left: object
    right: object

    def evaluate(self, index: Index) -> set[int]:
        return self.left.evaluate(index) | self.right.evaluate(index)


@dataclass(frozen=True)
class Not(_Q):
    child: object

    def evaluate(self, index: Index) -> set[int]:
        return set(index.doc_meta) - self.child.evaluate(index)


def _adjacent(prev: tuple[int, ...], nxt: tuple[int, ...]) -> tuple[int, ...]:
    i = j = 0
    found: list[int] = []
    while i < len(prev) and j < len(nxt):
        want = prev[i] + 1
        if nxt[j] == want:
            found.append(nxt[j])
            i += 1
            j += 1
        elif nxt[j] < want:
            j += 1
        else:
            i += 1
    return tuple(found)


def _chain(by_doc: list[dict[int, tuple[int, ...]]], doc_id: int) -> bool:
    chain = by_doc[0][doc_id]
    for col in by_doc[1:]:
        chain = _adjacent(chain, col[doc_id])
        if not chain:
            return False
    return True


def _tokens(query: str):
    i = 0
    n = len(query)
    while i < n:
        if query[i].isspace():
            i += 1
            continue
        if query[i] in "()":
            yield ("op", query[i])
            i += 1
            continue
        if query[i] == '"':
            j = query.find('"', i + 1)
            if j < 0:
                j = n
            yield ("phrase", query[i + 1 : j])
            i = j + 1
            continue
        j = i
        while j < n and not query[j].isspace() and query[j] not in '()\"':
            j += 1
        yield ("word", query[i:j])
        i = j


def _op(tok, *names: str) -> bool:
    return bool(tok) and tok[0] == "word" and tok[1].upper() in names


def _atom_node(tok):
    if tok[0] == "phrase":
        words = tuple(Tokenizer.tokenize(tok[1]))
        if len(words) > 1:
            return Phrase(words)
        return Term(words[0] if words else "")
    return Term(tok[1].lower())


def parse(query: str):
    tokens = list(_tokens(query))
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def eat():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        return tok

    def atom():
        tok = eat()
        if tok == ("op", "("):
            node = or_expr()
            if peek() == ("op", ")"):
                eat()
            return node
        return _atom_node(tok)

    def not_expr():
        # NOT binds only the next atom
        if _op(peek(), "NOT"):
            eat()
            return Not(atom())
        return atom()

    def and_expr():
        # a missing AND is still AND
        node = not_expr()
        while True:
            tok = peek()
            if tok is None or tok == ("op", ")") or _op(tok, "OR"):
                return node
            if _op(tok, "AND"):
                eat()
            node = node & not_expr()

    def or_expr():
        node = and_expr()
        while _op(peek(), "OR"):
            eat()
            node = node | and_expr()
        return node

    return or_expr()


def query_terms(node) -> list[str]:
    if isinstance(node, Term):
        return [node.value]
    if isinstance(node, Phrase):
        return list(node.words)
    if isinstance(node, Not):
        return []
    return query_terms(node.left) + query_terms(node.right)
