"""Input proteomes: identifier extraction, validation and staging.

Because SuppOrth now runs the predictors itself, it also controls their input.
Identifiers are read once from the FASTA files and reused to orient every pair
and to reject cross-species mistakes, instead of being inferred from whatever
columns a hand-exported table happened to contain.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from Bio import SeqIO

# Extension used for staged inputs; the only one all three predictors accept.
STAGED_SUFFIX = ".fasta"


@dataclass
class Proteome:
    label: str
    fasta: Path
    ids: frozenset[str]

    @property
    def size(self) -> int:
        return len(self.ids)


class ProteomeError(RuntimeError):
    pass


def read_ids(fasta: Path) -> frozenset[str]:
    """First whitespace-delimited token of each record header."""
    fasta = Path(fasta)
    if not fasta.exists():
        raise ProteomeError(f"FASTA not found: {fasta}")
    ids = {record.id.split()[0] for record in SeqIO.parse(str(fasta), "fasta")}
    if not ids:
        raise ProteomeError(f"no sequences parsed from {fasta}")
    return frozenset(ids)


def as_proteomes(
    first: Sequence[Proteome] | Proteome,
    *rest: Proteome,
) -> tuple[Proteome, ...]:
    """Accept either a sequence or the older ``sp1, sp2`` positional form."""
    if isinstance(first, Proteome):
        return (first, *rest)
    return tuple(first)


def load_proteomes(
    fastas: Sequence[Path],
    labels: Sequence[str] | None = None,
) -> tuple[Proteome, ...]:
    """Load two or more proteomes. Identifiers must be unique across the set."""
    paths = [Path(path).resolve() for path in fastas]
    if len(paths) < 2:
        raise ProteomeError("need at least two proteomes")

    names: list[str]
    if labels:
        names = [str(label).strip() for label in labels]
        if len(names) != len(paths):
            raise ProteomeError(
                f"--labels has {len(names)} name(s) but {len(paths)} FASTA files were given"
            )
        if any(not name for name in names):
            raise ProteomeError("labels must be non-empty")
    else:
        names = [_label(path) for path in paths]

    if len(set(names)) != len(names):
        raise ProteomeError(
            "proteome labels are not unique; pass explicit names with --labels"
        )

    proteomes = tuple(
        Proteome(name, path, read_ids(path)) for name, path in zip(names, paths)
    )

    seen: dict[str, str] = {}
    for proteome in proteomes:
        overlap = [gene for gene in proteome.ids if gene in seen]
        if overlap:
            sample = ", ".join(sorted(overlap)[:5])
            raise ProteomeError(
                f"{len(overlap)} identifier(s) occur in both {seen[overlap[0]]!r} "
                f"and {proteome.label!r} (e.g. {sample}). Orientation and paralogue "
                f"filtering depend on identifiers being unique per species; prefix "
                f"them per species and re-run."
            )
        for gene in proteome.ids:
            seen[gene] = proteome.label
    return proteomes


def load_pair(fasta1: Path, fasta2: Path, label1: str = "", label2: str = "") -> tuple[Proteome, Proteome]:
    labels = (label1, label2) if (label1 or label2) else None
    first, second = load_proteomes((fasta1, fasta2), labels)
    return first, second


def stage_input_dir(destination: Path, proteomes: Sequence[Proteome]) -> Path:
    """A directory holding the proteomes, as the predictors expect.

    All three predictors take a directory of FASTA files rather than named
    inputs, and they derive species names from file names. Staging with
    predictable names keeps their outputs parseable.

    The ``.fasta`` extension is not arbitrary: Broccoli scans its input
    directory for ``*.fasta`` specifically and reports an empty directory
    otherwise, while OrthoFinder and SonicParanoid accept it too.
    """
    destination.mkdir(parents=True, exist_ok=True)
    for existing in destination.glob("*.fa*"):
        existing.unlink()
    for proteome in proteomes:
        target = destination / f"{proteome.label}{STAGED_SUFFIX}"
        shutil.copyfile(proteome.fasta, target)
    return destination


def _label(fasta: Path) -> str:
    name = fasta.name
    for suffix in (".gz", ".faa", ".fa", ".fasta", ".pep", ".fas"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name or fasta.stem
