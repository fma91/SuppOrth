"""Identifier-driven parsing of predictor output tables.

The previous approach guessed which columns held gene identifiers, which broke
whenever a tool changed its output. Now that SuppOrth supplies the input FASTAs
it already knows every valid identifier, so a row can be read by asking which
tokens *are* known genes and ignoring everything else — scores, group numbers,
species names, bootstrap values and headers all fall away without needing a
rule per tool or per version.

The cost is that a row is read as "these species-1 genes relate to these
species-2 genes", losing any internal structure such as seed-orthologue
scoring. For a support-counting consensus that structure is not used anyway.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import AbstractSet, Iterator

_TOKEN_SPLIT = re.compile(r"[\t,;|\s]+")


def tokenize(line: str) -> list[str]:
    return [token.strip() for token in _TOKEN_SPLIT.split(line.strip()) if token.strip()]


def pairs_from_table(
    path: Path,
    sp1_ids: AbstractSet[str],
    sp2_ids: AbstractSet[str],
    *,
    max_row_pairs: int = 100_000,
) -> Iterator[tuple[str, str]]:
    """Yield cross-species pairs found on each row of a predictor table."""
    with Path(path).open(errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            tokens = tokenize(line)
            if not tokens:
                continue

            first = [t for t in tokens if t in sp1_ids]
            second = [t for t in tokens if t in sp2_ids]
            if not first or not second:
                continue
            if len(first) * len(second) > max_row_pairs:
                continue

            for left in first:
                for right in second:
                    yield left, right
