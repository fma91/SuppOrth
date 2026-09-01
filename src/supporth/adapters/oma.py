"""OMA standalone is not launched by SuppOrth.

OMA wants its own ``DB/`` of ``*.fa`` files, a Darwin runtime, and hours of
all-against-all alignment that the user already runs elsewhere — often on a
cluster, with more than two genomes. What SuppOrth needs is the pairwise
orthologue tables from that run, pointed at with ``--oma``.

The files live under ``Output/PairwiseOrthologs/``. Each pair is listed once.
Within-species files are not produced; extra species in a multi-genome run
are ignored because their identifiers are not in the two input proteomes.
"""

from __future__ import annotations

from pathlib import Path

from ..canonical import PredictorResult, build_result
from ..method_stack import OMA as OMA_STACK
from ..proteomes import Proteome
from .base import Adapter, AdapterError, MethodProfile, StageContext
from .tabular import pairs_from_table


class OMA(Adapter):
    name = "oma"
    label = "O"
    repo = ""
    default_ref = ""
    entrypoint_candidates = ()
    binaries = ()
    install_notes = (
        "not installed by suppOrth; pass --oma to an existing OMA standalone run"
    )

    def install(self, **kwargs):  # noqa: ANN003
        raise AdapterError(
            "OMA is not installed by suppOrth. Run OMA standalone yourself and "
            "pass the result directory with: suppOrth run ... --oma PATH"
        )

    def check(self, options: list[str] | None = None) -> tuple[bool, str]:
        return True, "user-supplied output; nothing to install"

    def method_profile(self, context: StageContext) -> MethodProfile:
        return MethodProfile(
            search=OMA_STACK.search,
            clustering=OMA_STACK.clustering,
            alignment=OMA_STACK.alignment,
            tree=OMA_STACK.tree,
            orthology=OMA_STACK.orthology,
        )

    def run(self, context: StageContext) -> Path:
        raise AdapterError(
            "OMA is not run by suppOrth; pass --oma PATH to an existing Output/ directory"
        )

    def parse(self, native_root: Path, sp1: Proteome, sp2: Proteome) -> PredictorResult:
        tables = self.result_tables(native_root, sp1=sp1, sp2=sp2)
        if not tables:
            raise AdapterError(
                f"no OMA pairwise tables under {native_root}. "
                f"Expected Output/PairwiseOrthologs/*.txt from an OMA standalone run."
            )
        raw: list[tuple[str, str]] = []
        for table in tables:
            raw.extend(pairs_from_table(table, sp1.ids, sp2.ids))
        return build_result(self.name, self.label, raw, sp1, sp2, tables)

    def result_tables(
        self,
        native_root: Path,
        sp1: Proteome | None = None,
        sp2: Proteome | None = None,
    ) -> list[Path]:
        root = Path(native_root).expanduser()
        if root.is_file():
            return [root]
        pairwise = self._pairwise_tables(root)
        if sp1 is not None and sp2 is not None:
            pairwise = self._tables_for_species(pairwise, sp1, sp2)
        if pairwise:
            return pairwise
        return sorted(p for p in root.rglob("OrthologousGroups.txt") if p.is_file())

    def _tables_for_species(
        self,
        tables: list[Path],
        sp1: Proteome,
        sp2: Proteome,
    ) -> list[Path]:
        """Keep the pairwise file for these two genomes when the name says so.

        An OMA run often includes extra species. Those files are large and
        contribute no pairs once identifiers are filtered, so they are skipped
        when a filename such as ``tmuris-contortus.txt`` matches both labels.
        """
        labels = {sp1.label.lower(), sp2.label.lower()}
        named = [
            path
            for path in tables
            if labels <= {part.lower() for part in path.stem.replace("_", "-").split("-")}
        ]
        return named or tables

    def _pairwise_tables(self, root: Path) -> list[Path]:
        directories = [p for p in root.rglob("PairwiseOrthologs") if p.is_dir()]
        if (root / "PairwiseOrthologs").is_dir():
            directories.append(root / "PairwiseOrthologs")
        found: list[Path] = []
        seen: set[Path] = set()
        for directory in directories:
            for path in directory.iterdir():
                if path in seen:
                    continue
                if not path.is_file() or path.suffix != ".txt":
                    continue
                if "cutted" in path.name.lower():
                    continue
                seen.add(path)
                found.append(path)
        return sorted(found)
