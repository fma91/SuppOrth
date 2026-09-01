"""OrthoFinder adapter.

Reference implementation for the adapter contract. OrthoFinder is the most
awkward of the three to wrap, for two reasons worth knowing about:

* it writes into a date-stamped ``Results_<Mmmdd>`` directory, and refuses to
  reuse an existing output directory, so output is located by search rather than
  by an assumed path;
* the pairwise orthologue tables live in ``Orthologues/`` and are the file we
  want, not ``Orthogroups.tsv`` — an orthogroup is a cluster, and expanding a
  cluster into all cross-species pairs is a weaker claim. The orthogroup table
  is used only as a fallback, and the result records which was read;
* its sequence search is single-threaded per job and parallelised by job count,
  which collapses to four jobs for a two-species comparison. See ``of_config``
  for how the thread count is restored.

OrthoFinder is GPL-licensed: it is fetched from upstream at install time and
invoked as a subprocess, never vendored into this repository.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..canonical import PredictorResult, build_result
from ..proteomes import Proteome
from .. import shell
from ..method_stack import ORTHOFINDER
from . import of_config
from .base import Adapter, AdapterError, MethodProfile, StageContext


def _split_genes(cell: str) -> list[str]:
    if not cell:
        return []
    return [token.strip() for token in cell.replace(";", ",").split(",") if token.strip()]


@dataclass
class Methods:
    """OrthoFinder's method choices, as configured for one run.

    The default is BLAST → MCL → MAFFT → IQ-TREE → gene-tree reconciliation
    (``-M msa -T iqtree``). That is the phylogenetic corner of the complementary
    split: Broccoli searches with DIAMOND and builds FastTree trees, and
    SonicParanoid searches with MMseqs2 and does not reconcile gene trees.
    ``dendroblast`` or ``-T fasttree`` are opt-in via ``--opt``.
    """

    search: str = ORTHOFINDER.search
    mode: str = "msa"
    aligner: str = ORTHOFINDER.alignment
    tree: str = ORTHOFINDER.tree

    @property
    def tree_program(self) -> str:
        if self.mode != "msa":
            return "dendroblast (distance trees from the similarity graph)"
        return self.tree or "iqtree"

    def required_binaries(self) -> tuple[str, ...]:
        needed = list(of_config.binaries_for_search(self.search or ORTHOFINDER.search))
        if self.mode != "msa":
            return tuple(needed)
        needed.append(self.aligner or ORTHOFINDER.alignment)
        tree = self.tree or ORTHOFINDER.tree
        # OrthoFinder spells FastTree lowercase in its config; the shared bin
        # carries the upstream capitalisation.
        needed.append("FastTree" if tree.lower() == "fasttree" else tree)
        return tuple(needed)


def parse_methods(options: list[str] | None) -> Methods:
    """Read OrthoFinder's -S/-M/-A/-T flags out of pass-through options."""
    methods = Methods()
    tokens = list(options or [])
    flags = {"-S": "search", "-M": "mode", "-A": "aligner", "-T": "tree"}
    for index, token in enumerate(tokens[:-1]):
        attribute = flags.get(token)
        if attribute:
            setattr(methods, attribute, tokens[index + 1])
    return methods


class OrthoFinder(Adapter):
    name = "orthofinder"
    label = "F"
    repo = "https://github.com/davidemms/OrthoFinder.git"
    default_ref = "v2.5.5"
    entrypoint_candidates = (
        "orthofinder.py",
        "orthofinder/orthofinder.py",
        "orthofinder/__main__.py",
        "scripts_of/orthofinder.py",
        "*/orthofinder.py",
    )
    binaries = ("mcl",)
    # scripts_of/__main__.py prepends its own bin/ to PATH, so its bundled
    # diamond, mcl and fastme win over the shared ones unless they are linked.
    shadows_path = True
    install_notes = (
        "GPL-3.0; fetched from upstream and invoked as a subprocess. "
        "Default stack is BLAST, MCL, MAFFT and IQ-TREE (gene-tree "
        "reconciliation). Bundled helper binaries are linked to the shared bin "
        "at install time."
    )

    search_program = ORTHOFINDER.search
    tree_method = "msa"

    def post_install(self, root: Path, *, interpreter: Path | None = None, log=print) -> None:
        """Give OrthoFinder its own interpreter.

        It imports numpy and scipy at start-up. Those are heavy and version
        sensitive, so they go in a private virtualenv rather than becoming
        dependencies of SuppOrth itself.
        """
        self.build_venv(root, interpreter, ("numpy", "scipy"), log=log)

    def _interpreter(self, root: Path) -> str:
        return str(self.venv_python(root) or self.python())

    def check(self, options: list[str] | None = None) -> tuple[bool, str]:
        ready, detail = super().check(options)
        if not ready:
            return ready, detail
        root = self.installed_root()
        probe = shell.run(
            [self._interpreter(root), "-c", "import numpy, scipy"],
            check=False,
        )
        if probe.returncode != 0:
            return False, "numpy/scipy unavailable; re-run suppOrth install --tools orthofinder"
        return True, detail

    def extra_binaries(self, options: list[str] | None = None) -> tuple[str, ...]:
        return parse_methods(options).required_binaries()

    def method_profile(self, context: StageContext) -> MethodProfile:
        methods = parse_methods(context.options)
        program = methods.search or self.search_program
        jobs = of_config.search_job_count(2)
        return MethodProfile(
            search=f"{program} (via OrthoFinder)",
            clustering="MCL on the BLAST graph",
            alignment=methods.aligner or ("none" if methods.mode != "msa" else ORTHOFINDER.alignment),
            tree=methods.tree_program,
            orthology=ORTHOFINDER.orthology if methods.mode == "msa" else "dendroblast graph clustering",
            extra={
                "mode": methods.mode,
                "search_jobs": str(jobs),
                "threads_per_search": str(
                    of_config.threads_per_job(context.threads, jobs)
                ),
            },
        )

    def run(self, context: StageContext) -> Path:
        root = self.installed_root()
        entrypoint = self.locate_entrypoint(root)

        # OrthoFinder aborts if the results directory already exists.
        output = context.work_dir / "results"
        if output.exists():
            shutil.rmtree(output)
        context.work_dir.mkdir(parents=True, exist_ok=True)

        methods = parse_methods(context.options)
        requested = methods.search or self.search_program

        jobs = of_config.search_job_count(2)
        per_job = of_config.threads_per_job(context.threads, jobs)

        with of_config.config_lock(root):
            search_program = self._configure_search(root, requested, per_job)

            argv = [
                self._interpreter(root),
                str(entrypoint),
                "-f",
                str(context.input_dir),
                "-o",
                str(output),
                "-t",
                str(context.threads),
                "-a",
                str(max(1, context.threads // 2)),
                "-S",
                search_program,
            ]
            passthrough = of_config.strip_option(context.options, "-S")
            # Defaults are only supplied when the run did not override them, so
            # a pass-through "-M dendroblast" does not collide with MSA mode.
            if "-M" not in set(passthrough):
                argv += ["-M", self.tree_method]
            if methods.mode == "msa":
                if "-A" not in set(passthrough):
                    argv += ["-A", methods.aligner or ORTHOFINDER.alignment]
                if "-T" not in set(passthrough):
                    argv += ["-T", methods.tree or ORTHOFINDER.tree]
            argv += passthrough

            # OrthoFinder runs its search jobs concurrently, so the OpenMP limit
            # is the per-job share rather than the whole machine.
            result = shell.run(
                argv,
                cwd=context.work_dir,
                log=context.log,
                timeout=None,
                threads=per_job,
                check=False,
            )

        if result.returncode != 0:
            self._salvage_or_fail(output, argv, result.returncode, context)
        return output

    def _salvage_or_fail(self, output: Path, argv, returncode: int, context: StageContext) -> None:
        """Keep usable output from a run that ended badly.

        OrthoFinder can complete the search and orthogroup stages and then fail
        during tree inference — on small datasets it raises an UnboundLocalError
        there and reports it as running out of memory. The orthogroups it already
        wrote are still worth having, so a non-zero exit is only fatal when
        nothing parseable was produced.
        """
        salvageable = self._pairwise_tables(output) or self._orthogroups_table(output)
        if not salvageable:
            raise shell.CommandFailed(argv, returncode, context.log)
        with context.log.open("a") as handle:
            handle.write(
                f"\n# SuppOrth: OrthoFinder exited {returncode} but left parseable "
                f"output; continuing with what it produced.\n"
            )

    def _configure_search(self, root: Path, program: str, threads_per_search: int) -> str:
        """Register a thread-aware copy of the search method and return its name."""
        config = of_config.find_config(root)
        if config is None:
            return program
        return of_config.register_threaded_search(config, program, threads_per_search)

    def parse(self, native_root: Path, sp1: Proteome, sp2: Proteome) -> PredictorResult:
        pairwise = self._pairwise_tables(native_root)
        note = ""
        if pairwise:
            raw, sources = self._read_pairwise(pairwise, sp1, sp2)
        else:
            table = self._orthogroups_table(native_root)
            if table is None:
                raise AdapterError(
                    f"OrthoFinder produced no parseable output under {native_root}. "
                    f"Expected Orthologues/*__v__*.tsv or Orthogroups/Orthogroups.tsv; "
                    f"check the stage log."
                )
            raw, sources = self._read_orthogroups(table, sp1, sp2)
            note = (
                "orthogroup fallback: pairs expanded from clusters, not from "
                "OrthoFinder's pairwise orthologue tables"
            )

        result = build_result(self.name, self.label, raw, sp1, sp2, sources)
        result.note = note
        return result

    # ---- output discovery ---------------------------------------------

    def _pairwise_tables(self, native_root: Path) -> list[Path]:
        found = [p for p in native_root.rglob("*__v__*.tsv") if p.is_file()]
        preferred = [p for p in found if "Orthologues" in p.parts]
        return preferred or found

    def _orthogroups_table(self, native_root: Path) -> Path | None:
        for candidate in native_root.rglob("Orthogroups.tsv"):
            if candidate.is_file():
                return candidate
        for candidate in native_root.rglob("Orthogroups.csv"):
            if candidate.is_file():
                return candidate
        return None

    # ---- readers -------------------------------------------------------

    def _read_pairwise(
        self, tables: list[Path], sp1: Proteome, sp2: Proteome
    ) -> tuple[list[tuple[str, str]], list[Path]]:
        raw: list[tuple[str, str]] = []
        used: list[Path] = []
        wanted = {sp1.label, sp2.label}

        for table in tables:
            stem = table.stem
            if "__v__" in stem:
                left_species, right_species = stem.split("__v__", 1)
                if wanted and {left_species, right_species} != wanted:
                    continue
            used.append(table)
            with table.open(newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader, None)
                if header is None:
                    continue
                for row in reader:
                    if len(row) < 3:
                        continue
                    left_genes = _split_genes(row[1])
                    right_genes = _split_genes(row[2])
                    for left in left_genes:
                        for right in right_genes:
                            raw.append((left, right))
        return raw, used

    def _read_orthogroups(
        self, table: Path, sp1: Proteome, sp2: Proteome
    ) -> tuple[list[tuple[str, str]], list[Path]]:
        raw: list[tuple[str, str]] = []
        delimiter = "\t" if table.suffix == ".tsv" else ","
        with table.open(newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            header = next(reader, None)
            if header is None:
                return raw, [table]
            for row in reader:
                genes = [gene for cell in row[1:] for gene in _split_genes(cell)]
                first = [g for g in genes if g in sp1.ids]
                second = [g for g in genes if g in sp2.ids]
                for left in first:
                    for right in second:
                        raw.append((left, right))
        return raw, [table]
