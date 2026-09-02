"""One canonical representation of a predictor's result.

Every adapter converts its tool's native output into the same object: a set of
ordered ``(protein_a, protein_b)`` pairs, with the protein from the earlier
input FASTA first. Downstream code therefore never needs to know which tool
produced a result, and the old split between pair-tuple sets and query-keyed
dictionaries disappears.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .proteomes import Proteome, as_proteomes

Pair = tuple[str, str]


@dataclass
class PredictorResult:
    tool: str
    label: str
    pairs: set[Pair] = field(default_factory=set)
    dropped_unknown: int = 0
    dropped_same_species: int = 0
    source_files: list[str] = field(default_factory=list)
    # Set when the result is weaker than the tool's best output, e.g. expanded
    # from clusters because the pairwise tables were not produced.
    note: str = ""

    def summary(self) -> str:
        text = (
            f"{self.tool}: {len(self.pairs)} pairs "
            f"(dropped {self.dropped_unknown} unknown id, "
            f"{self.dropped_same_species} same-species)"
        )
        return f"{text} [{self.note}]" if self.note else text


def build_result(
    tool: str,
    label: str,
    raw_pairs: Iterable[Pair],
    proteomes: Sequence[Proteome] | Proteome,
    *rest: Proteome | Iterable[Path],
    source_files: Iterable[Path] = (),
) -> PredictorResult:
    """Orient and filter raw pairs against the input proteomes.

    Pairs are stored with the earlier input FASTA first. Within-species pairs
    are dropped: those are the tools reporting paralogues, which is a different
    relationship from the cross-species orthology this consensus is about.

    ``build_result(..., sp1, sp2)`` and ``build_result(..., sp1, sp2, tables)``
    remain valid; a sequence of proteomes is preferred for *n*-species runs.
    """
    extras: list[Proteome] = []
    for item in rest:
        if isinstance(item, Proteome):
            extras.append(item)
        else:
            source_files = item
            break
    loaded = as_proteomes(proteomes, *extras)

    owner: dict[str, int] = {}
    for index, proteome in enumerate(loaded):
        for gene in proteome.ids:
            owner[gene] = index

    result = PredictorResult(
        tool=tool,
        label=label,
        source_files=[str(p) for p in source_files],
    )
    for left, right in raw_pairs:
        left_i = owner.get(left)
        right_i = owner.get(right)
        if left_i is None or right_i is None:
            result.dropped_unknown += 1
            continue
        if left_i == right_i:
            result.dropped_same_species += 1
            continue
        result.pairs.add((left, right) if left_i < right_i else (right, left))
    return result


def write_json(result: PredictorResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tool": result.tool,
        "label": result.label,
        "n_pairs": len(result.pairs),
        "dropped_unknown": result.dropped_unknown,
        "dropped_same_species": result.dropped_same_species,
        "source_files": result.source_files,
        "note": result.note,
        "pairs": sorted(list(pair) for pair in result.pairs),
    }
    with path.open("w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")
    return path


def read_json(path: Path) -> PredictorResult:
    with Path(path).open() as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "pairs" in payload:
        return PredictorResult(
            tool=payload.get("tool", Path(path).stem),
            label=payload.get("label", Path(path).stem[:1].upper()),
            pairs={(a, b) for a, b in payload["pairs"]},
            dropped_unknown=payload.get("dropped_unknown", 0),
            dropped_same_species=payload.get("dropped_same_species", 0),
            source_files=payload.get("source_files", []),
            note=payload.get("note", ""),
        )

    # Legacy layout: {query_id: [orthologue_id, ...]}
    pairs: set[Pair] = set()
    for query, orthologues in payload.items():
        for orthologue in orthologues or []:
            pairs.add((query, orthologue))
    stem = Path(path).stem
    return PredictorResult(tool=stem, label=stem[:1].upper(), pairs=pairs)


def write_legacy_json(result: PredictorResult, path: Path) -> Path:
    """Query-keyed form, for compatibility with the pre-0.4 scripts."""
    grouped: dict[str, list[str]] = {}
    for left, right in sorted(result.pairs):
        grouped.setdefault(left, []).append(right)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(grouped, handle, indent=1)
        handle.write("\n")
    return path
