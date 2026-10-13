from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator

# [^\W_]+  = letters/digits; (?:['’ʼ-][^\W_]+)* = optional ' or - chunks inside word
# don't → don't | good-for-nothing → good-for-nothing | 4 → 4 | Hello, world! → hello  world
regex = re.compile(r"[^\W_]+(?:[''\u2019\u02bc\-][^\W_]+)*", re.UNICODE)


class Tokenizer:
    @staticmethod
    def tokenize(text: str) -> Iterator[str]:
        text = unicodedata.normalize("NFKC", text).lower()
        for m in regex.finditer(text):
            yield m.group(0)
