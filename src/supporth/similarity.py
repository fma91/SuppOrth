"""Alignment statistics used to annotate the consensus table.

This is annotation, not evidence: the search runs after the predictors and does
not add or remove pairs. It exists so a reader can see whether a
weakly-supported pair is at least a decent alignment.

Both directions are searched and the better hit per unordered pair is kept,
because query/subject asymmetry is an artefact of the search rather than a
property of the relationship. A pair with no hit is reported as missing rather
than as zero identity: those are different statements, and the old behaviour of
writing 0.0 made unaligned pairs look like terrible alignments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import shell
from .proteomes import Proteome

OUTFMT = ("qseqid", "sseqid", "pident", "evalue", "bitscore")


@dataclass
class Hit:
    identity: float
    evalue: float
    bitscore: float


class SimilarityIndex:
    """Best alignment per unordered protein pair."""

    def __init__(self) -> None:
        self._hits: dict[tuple[str, str], Hit] = {}

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def add(self, query: str, subject: str, hit: Hit) -> None:
        key = self._key(query, subject)
        existing = self._hits.get(key)
        if existing is None or hit.bitscore > existing.bitscore:
            self._hits[key] = hit

    def get(self, a: str, b: str) -> Hit | None:
        return self._hits.get(self._key(a, b))

    def __len__(self) -> int:
        return len(self._hits)

    def load_tabular(self, path: Path) -> None:
        with Path(path).open(errors="replace") as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 5:
                    continue
                query, subject = fields[0], fields[1]
                try:
                    hit = Hit(float(fields[2]), float(fields[3]), float(fields[4]))
                except ValueError:
                    continue
                self.add(query, subject, hit)


def index_from_orthofinder(
    work_dir: Path,
    sp1: Proteome,
    sp2: Proteome,
    *,
    log=print,
) -> SimilarityIndex:
    """Reuse the all-vs-all search OrthoFinder has already performed.

    OrthoFinder keeps its raw alignment tables in ``WorkingDirectory``, keyed
    by internal identifiers such as ``0_1474``. Reading them takes seconds
    where repeating the search on a large pair takes hours, so an existing run
    is worth adopting when one is at hand.

    Only the cross-species tables are read. The within-species ones describe
    paralogues, which the consensus table has no column for.
    """
    work_dir = Path(work_dir).expanduser()
    if (work_dir / "WorkingDirectory").is_dir():
        work_dir = work_dir / "WorkingDirectory"

    species = _orthofinder_species(work_dir / "SpeciesIDs.txt")
    wanted = {}
    for index, filename in species.items():
        for proteome in (sp1, sp2):
            if filename == proteome.fasta.name:
                wanted[index] = proteome
    if len(wanted) < 2:
        raise ValueError(
            f"{work_dir / 'SpeciesIDs.txt'} lists {sorted(species.values())}, which does "
            f"not cover {sp1.fasta.name} and {sp2.fasta.name}"
        )

    names = _orthofinder_sequence_names(work_dir / "SequenceIDs.txt")
    index = SimilarityIndex()
    for first in sorted(wanted):
        for second in sorted(wanted):
            if first == second:
                continue
            table = work_dir / f"Blast{first}_{second}.txt"
            if not table.is_file():
                continue
            log(f"  reading {table.name}")
            _load_orthofinder_table(table, names, index)
    return index


def _orthofinder_species(path: Path) -> dict[str, str]:
    species: dict[str, str] = {}
    with Path(path).open(errors="replace") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, filename = line.split(":", 1)
            species[key.strip()] = filename.strip()
    return species


def _orthofinder_sequence_names(path: Path) -> dict[str, str]:
    """Internal ``species_sequence`` identifiers mapped to real gene names."""
    names: dict[str, str] = {}
    with Path(path).open(errors="replace") as handle:
        for line in handle:
            if ":" not in line:
                continue
            key, description = line.split(":", 1)
            fields = description.split()
            if fields:
                names[key.strip()] = fields[0]
    return names


def _load_orthofinder_table(
    path: Path,
    names: dict[str, str],
    index: SimilarityIndex,
) -> None:
    with Path(path).open(errors="replace") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 12:
                continue
            query = names.get(fields[0])
            subject = names.get(fields[1])
            if query is None or subject is None:
                continue
            # BLAST tabular format 6, whose last two columns are the e-value and
            # the bit score. Read from the end so a table carrying extra columns
            # is still understood.
            try:
                hit = Hit(float(fields[2]), float(fields[-2]), float(fields[-1]))
            except ValueError:
                continue
            index.add(query, subject, hit)


def build_index(
    sp1: Proteome,
    sp2: Proteome,
    work_dir: Path,
    *,
    threads: int = 1,
    program: str = "diamond",
    evalue: float = 1e-3,
    max_targets: int = 10,
    log=print,
) -> SimilarityIndex:
    work_dir.mkdir(parents=True, exist_ok=True)
    index = SimilarityIndex()

    if program == "none":
        return index

    runner = _diamond if program == "diamond" else _blastp
    for query, subject, tag in ((sp1, sp2, "fwd"), (sp2, sp1, "rev")):
        output = work_dir / f"{tag}.tsv"
        if not output.exists():
            log(f"  {program} {query.label} vs {subject.label}")
            runner(query, subject, work_dir, output, threads, evalue, max_targets, tag)
        index.load_tabular(output)
    return index


def _diamond(
    query: Proteome,
    subject: Proteome,
    work_dir: Path,
    output: Path,
    threads: int,
    evalue: float,
    max_targets: int,
    tag: str,
) -> None:
    binary = shell.which("diamond")
    if binary is None:
        raise RuntimeError("diamond not available in the shared bin")

    database = work_dir / f"{tag}.dmnd"
    shell.run(
        [
            binary,
            "makedb",
            "--in",
            str(subject.fasta),
            "-d",
            str(database),
            "--threads",
            str(threads),
            "--quiet",
        ],
        log=work_dir / f"{tag}.makedb.log",
        threads=threads,
    )
    shell.run(
        [
            binary,
            "blastp",
            "-q",
            str(query.fasta),
            "-d",
            str(database),
            "-o",
            str(output),
            "--outfmt",
            "6",
            *OUTFMT,
            "--evalue",
            str(evalue),
            "--max-target-seqs",
            str(max_targets),
            "--threads",
            str(threads),
            "--quiet",
        ],
        log=work_dir / f"{tag}.search.log",
        threads=threads,
    )


def _blastp(
    query: Proteome,
    subject: Proteome,
    work_dir: Path,
    output: Path,
    threads: int,
    evalue: float,
    max_targets: int,
    tag: str,
) -> None:
    makedb = shell.which("makeblastdb")
    blastp = shell.which("blastp")
    if makedb is None or blastp is None:
        raise RuntimeError("blastp/makeblastdb not available in the shared bin")

    database = work_dir / f"{tag}_db"
    shell.run(
        [makedb, "-in", str(subject.fasta), "-dbtype", "prot", "-out", str(database)],
        log=work_dir / f"{tag}.makedb.log",
    )
    shell.run(
        [
            blastp,
            "-query",
            str(query.fasta),
            "-db",
            str(database),
            "-out",
            str(output),
            "-outfmt",
            "6 " + " ".join(OUTFMT),
            "-evalue",
            str(evalue),
            "-max_target_seqs",
            str(max_targets),
            "-num_threads",
            str(threads),
        ],
        log=work_dir / f"{tag}.search.log",
        threads=threads,
    )
