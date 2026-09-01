"""Input proteomes: identifier extraction, validation and staging.

Because SuppOrth now runs the predictors itself, it also controls their input.
Identifiers are read once from the FASTA files and reused to orient every pair
and to reject cross-species mistakes, instead of being inferred from whatever
columns a hand-exported table happened to contain.
"""

from __future__ import annotations

import shutil
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


def load_pair(fasta1: Path, fasta2: Path, label1: str = "", label2: str = "") -> tuple[Proteome, Proteome]:
    fasta1, fasta2 = Path(fasta1).resolve(), Path(fasta2).resolve()
    first = Proteome(label1 or _label(fasta1), fasta1, read_ids(fasta1))
    second = Proteome(label2 or _label(fasta2), fasta2, read_ids(fasta2))

    if first.label == second.label:
        raise ProteomeError(
            f"both proteomes resolve to the label {first.label!r}; "
            f"pass explicit labels with --labels"
        )

    shared = first.ids & second.ids
    if shared:
        sample = ", ".join(sorted(shared)[:5])
        raise ProteomeError(
            f"{len(shared)} identifier(s) occur in both proteomes (e.g. {sample}). "
            f"Orientation and paralogue filtering depend on identifiers being "
            f"unique per species; prefix them per species and re-run."
        )
    return first, second


def stage_input_dir(destination: Path, proteomes: tuple[Proteome, Proteome]) -> Path:
    """A directory holding exactly the two proteomes, as the predictors expect.

    All three predictors take a directory of FASTA files rather than two named
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
