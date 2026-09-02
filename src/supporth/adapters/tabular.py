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
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import AbstractSet, Iterator

# Do not split on "|": NCBI identifiers such as gi|3844770|gb|AAC71393.1|
# would otherwise never match the FASTA ids.
_TOKEN_SPLIT = re.compile(r"[\t,;\s]+")


def tokenize(line: str) -> list[str]:
    return [token.strip() for token in _TOKEN_SPLIT.split(line.strip()) if token.strip()]


def pairs_from_table(
    path: Path,
    *id_sets: AbstractSet[str],
    max_row_pairs: int = 100_000,
) -> Iterator[tuple[str, str]]:
    """Yield cross-species pairs found on each row of a predictor table.

    Two identifier sets recover the original two-proteome behaviour. More than
    two yield every cross-species combination on the row, oriented by the order
    of the sets (the same order as the input FASTAs).
    """
    if len(id_sets) < 2:
        raise ValueError("pairs_from_table needs at least two identifier sets")

    owner: dict[str, int] = {}
    for index, ids in enumerate(id_sets):
        for gene in ids:
            owner[gene] = index

    with Path(path).open(errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            tokens = tokenize(line)
            if not tokens:
                continue

            by_species: dict[int, list[str]] = {}
            for token in tokens:
                index = owner.get(token)
                if index is None:
                    continue
                by_species.setdefault(index, []).append(token)
            if len(by_species) < 2:
                continue

            n_pairs = 0
            species = sorted(by_species)
            for left_i, right_i in combinations(species, 2):
                n_pairs += len(by_species[left_i]) * len(by_species[right_i])
            if n_pairs > max_row_pairs:
                continue

            for left_i, right_i in combinations(species, 2):
                for left in by_species[left_i]:
                    for right in by_species[right_i]:
                        yield left, right


def id_sets(proteomes: Sequence) -> tuple[frozenset[str], ...]:
    return tuple(proteome.ids for proteome in proteomes)
