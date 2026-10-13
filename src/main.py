from __future__ import annotations

import logging

from findex.index import main as build_index
from findex.score import sanity
from findex.search import main as search_index
from findex.search import precision_at_5


def main() -> None:
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    index = build_index()
    sanity(index)
    precision_at_5(index)
    search_index()


if __name__ == "__main__":
    main()
