"""Broccoli adapter.

Broccoli is distributed as a script rather than a package: it is run from a
working directory and writes numbered ``dir_step*`` folders into it, so each
stage gets its own scratch directory. Its pairwise output is produced by
step 4.

Its phylogeny-aware network clustering, run on a DIAMOND search, is the graph
corner of the complementary split: OrthoFinder searches with BLAST and
reconciles gene trees, SonicParanoid searches with MMseqs2.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..canonical import PredictorResult, build_result
from ..proteomes import Proteome
from ..method_stack import BROCCOLI
from .. import shell
from .base import Adapter, AdapterError, MethodProfile, StageContext
from .tabular import id_sets, pairs_from_table


class Broccoli(Adapter):
    name = "broccoli"
    label = "B"
    repo = "https://github.com/rderelle/Broccoli.git"
    default_ref = "v1.4.0"
    entrypoint_candidates = ("broccoli.py", "*/broccoli.py")
    binaries = ("diamond", "FastTree")
    install_notes = (
        "run as a script from a per-stage working directory; "
        "needs diamond and FastTree from the shared bin, plus ete3"
    )

    def post_install(self, root: Path, *, interpreter: Path | None = None, log=print) -> None:
        self.build_venv(root, interpreter, ("ete3", "six", "numpy"), log=log)

    def _interpreter(self, root: Path) -> str:
        return str(self.venv_python(root) or self.python())

    def method_profile(self, context: StageContext) -> MethodProfile:
        return MethodProfile(
            search="DIAMOND (via Broccoli)",
            clustering=BROCCOLI.clustering,
            alignment=BROCCOLI.alignment,
            tree="FastTree, per-orthogroup",
            orthology=BROCCOLI.orthology,
        )

    def run(self, context: StageContext) -> Path:
        root = self.installed_root()
        entrypoint = self.locate_entrypoint(root)

        output = context.work_dir / "results"
        output.mkdir(parents=True, exist_ok=True)

        argv = [
            self._interpreter(root),
            str(entrypoint),
            "-dir",
            str(context.input_dir),
            "-threads",
            str(context.threads),
            *context.options,
        ]
        shell.run(argv, cwd=output, log=context.log, timeout=None, threads=context.threads)
        return output

    def parse(self, native_root: Path, proteomes: Sequence[Proteome] | Proteome, *rest: Proteome) -> PredictorResult:
        loaded = (proteomes, *rest) if isinstance(proteomes, Proteome) else tuple(proteomes)
        tables = self._result_tables(native_root)
        if not tables:
            raise AdapterError(
                f"Broccoli produced no pair table under {native_root}. "
                f"Expected dir_step4/orthologous_pairs.txt; check the stage log."
            )
        raw: list[tuple[str, str]] = []
        for table in tables:
            raw.extend(pairs_from_table(table, *id_sets(loaded)))
        return build_result(self.name, self.label, raw, loaded, source_files=tables)

    def _result_tables(self, native_root: Path) -> list[Path]:
        pairs = [p for p in native_root.rglob("*orthologous_pairs*") if p.is_file()]
        if pairs:
            return pairs
        return [p for p in native_root.rglob("*orthologous_groups*") if p.is_file()]
