from __future__ import annotations

import tracemalloc as memory_traker
from collections import Counter
from time import perf_counter

from findex.documents import CorpusReader
from findex.models import StatsResult
from findex.tokenizer import Tokenizer


class LightStats:
    def __init__(self, reader: CorpusReader, tokenizer: Tokenizer) -> None:
        self.reader = reader
        self.tokenizer = tokenizer

    def run(self) -> StatsResult:
        memory_traker.start()
        t0 = perf_counter()
        counts: Counter[str] = Counter()
        n_docs = 0
        n_tokens = 0

        for text in self.reader.iter_documents():
            n_docs += 1
            # yields tokens one by one from text
            for tok in self.tokenizer.tokenize(text):
                counts[tok] += 1
                n_tokens += 1

        # _ = current, we do not need it
        _, peak = memory_traker.get_traced_memory()
        memory_traker.stop()
        return StatsResult(
            n_docs=n_docs,
            n_tokens=n_tokens,
            vocab_size=len(counts),
            top50=counts.most_common(50),
            elapsed_s=perf_counter() - t0,
            peak_kib=peak / 1024,
        )


class HeavyStats:
    def __init__(self, reader: CorpusReader, tokenizer: Tokenizer) -> None:
        self.reader = reader
        self.tokenizer = tokenizer

    def run(self) -> StatsResult:
        memory_traker.start()
        t0 = perf_counter()

        documents = list(self.reader.iter_documents())
        all_tokens: list[str] = []
        for text in documents:
            all_tokens.extend(self.tokenizer.tokenize(text))

        counts = Counter(all_tokens)
        elapsed = perf_counter() - t0
        _, peak = memory_traker.get_traced_memory()
        memory_traker.stop()
        return StatsResult(
            n_docs=len(documents),
            n_tokens=len(all_tokens),
            vocab_size=len(counts),
            top50=counts.most_common(50),
            elapsed_s=elapsed,
            peak_kib=peak / 1024,
        )
