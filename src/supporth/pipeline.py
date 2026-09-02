"""Sequential execution of the predictors and assembly of the consensus.

The predictors run one after another rather than in parallel. Each of them
already saturates the machine and expects to manage its own threads, so
overlapping them mostly produces contention and unreliable timings.

Two consequences of long runtimes shape this module: every predictor is checked
for readiness *before* the first one starts, and every completed stage writes
its canonical result to disk so ``--resume`` never repeats finished work.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from . import canonical, consensus, deps, figures, manifest, paths, similarity, shell
from .adapters import Adapter, StageContext, resolve
from .adapters.oma import OMA
from .proteomes import Proteome, load_proteomes, stage_input_dir


class PipelineError(RuntimeError):
    pass


@dataclass
class RunConfig:
    outdir: Path
    fastas: tuple[Path, ...] = ()
    fasta1: Path | None = None
    fasta2: Path | None = None
    tools: str = "all"
    labels: tuple[str, ...] | None = None
    threads: int = 0
    annotate: str = "diamond"
    resume: bool = False
    support_lv: int = 3
    identity_filter: float = 80.0
    collapse: tuple[Path, Path] | None = None
    tool_options: dict[str, list[str]] = field(default_factory=dict)
    oma: Path | None = None

    def __post_init__(self) -> None:
        if self.fastas:
            self.fastas = tuple(Path(path) for path in self.fastas)
        elif self.fasta1 is not None and self.fasta2 is not None:
            self.fastas = (Path(self.fasta1), Path(self.fasta2))
        if len(self.fastas) < 2:
            raise ValueError("need at least two FASTA files")


@dataclass
class StageReport:
    tool: str
    label: str
    status: str
    seconds: float
    n_pairs: int
    method: dict
    commit: str
    dropped_unknown: int = 0
    dropped_same_species: int = 0
    note: str = ""
    error: str = ""


def run_pipeline(config: RunConfig, log: Callable[[str], None] = print) -> dict:
    started = time.monotonic()
    paths.ensure_layout()

    outdir = Path(config.outdir).resolve()
    (outdir / "pairs").mkdir(parents=True, exist_ok=True)
    work_root = outdir / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    proteomes = load_proteomes(config.fastas, config.labels)
    for index, proteome in enumerate(proteomes, start=1):
        log(f"Species {index}: {proteome.label} ({proteome.size} proteins)")

    adapters = resolve(config.tools)
    threads = shell.cpu_count(config.threads or None)

    _preflight(adapters, config.tool_options, log)

    input_dir = stage_input_dir(work_root / "input", proteomes)

    reports: list[StageReport] = []
    results: list[canonical.PredictorResult] = []

    for index, adapter in enumerate(adapters, start=1):
        log(f"\n[{index}/{len(adapters)}] {adapter.name}")
        pairs_json = outdir / "pairs" / f"{adapter.name}.json"

        if config.resume and pairs_json.exists():
            result = canonical.read_json(pairs_json)
            results.append(result)
            reports.append(
                StageReport(
                    tool=adapter.name,
                    label=adapter.label,
                    status="reused",
                    seconds=0.0,
                    n_pairs=len(result.pairs),
                    method={},
                    commit=_commit(adapter),
                )
            )
            log(f"  reusing {pairs_json.name} ({len(result.pairs)} pairs)")
            continue

        context = StageContext(
            input_dir=input_dir,
            work_dir=work_root / adapter.name,
            proteomes=proteomes,
            threads=threads,
            options=config.tool_options.get(adapter.name, []),
        )
        context.work_dir.mkdir(parents=True, exist_ok=True)

        stage_started = time.monotonic()
        try:
            native_root = adapter.run(context)
            result = adapter.parse(native_root, proteomes)
        except Exception as error:  # noqa: BLE001 - one predictor must not sink the run
            elapsed = time.monotonic() - stage_started
            log(f"  FAILED after {elapsed / 60:.1f} min: {error}")
            log(f"  log: {context.log}")
            reports.append(
                StageReport(
                    tool=adapter.name,
                    label=adapter.label,
                    status="failed",
                    seconds=elapsed,
                    n_pairs=0,
                    method=adapter.method_profile(context).as_dict(),
                    commit=_commit(adapter),
                    error=str(error),
                )
            )
            continue

        elapsed = time.monotonic() - stage_started
        canonical.write_json(result, pairs_json)
        canonical.write_legacy_json(result, outdir / "pairs" / f"{adapter.name}_result_dict.json")
        results.append(result)
        reports.append(
            StageReport(
                tool=adapter.name,
                label=adapter.label,
                status="ok",
                seconds=elapsed,
                n_pairs=len(result.pairs),
                method=adapter.method_profile(context).as_dict(),
                commit=_commit(adapter),
                dropped_unknown=result.dropped_unknown,
                dropped_same_species=result.dropped_same_species,
                note=result.note,
            )
        )
        log(f"  {result.summary()} in {elapsed / 60:.1f} min")

    if config.oma:
        oma_result, oma_report = _ingest_oma(config, proteomes, outdir, log)
        if oma_result is not None:
            results.append(oma_result)
            reports.append(oma_report)

    if not results:
        raise PipelineError(
            "no predictor produced a usable result; see the stage logs under "
            f"{work_root}"
        )

    log("\nAnnotating with alignment statistics")
    index = similarity.build_index(
        proteomes,
        work_root / "similarity",
        threads=threads,
        program=config.annotate,
        log=log,
    )
    log(f"  {len(index)} aligned pairs available for annotation")

    table = consensus.build_table(results, index, proteomes)
    unfiltered = len(table)
    table = consensus.filter_table(
        table,
        support_lv=config.support_lv,
        identity_filter=config.identity_filter,
    )
    log(
        f"  kept {len(table)} of {unfiltered} pairs "
        f"(Lv_support ≥ {config.support_lv} or Identity > {config.identity_filter})"
    )

    output_tsv = outdir / "suppOrth.tsv"
    table.to_csv(output_tsv, sep="\t", index=False)

    summary = consensus.agreement_summary(table)
    summary.to_csv(outdir / "agreement_summary.tsv", sep="\t", index=False)

    figure_path = figures.write_support_figure(
        results, outdir / "suppOrth.png", log=log
    )

    collapsed_path = None
    if config.collapse:
        if len(proteomes) != 2:
            raise PipelineError("--collapse is only defined for exactly two proteomes")
        collapsed = consensus.collapse_to_genes(table, *config.collapse)
        collapsed_path = outdir / "suppOrth.genes.tsv"
        collapsed.to_csv(collapsed_path, sep="\t", index=False)
        log(f"Collapsed to {len(collapsed)} gene pairs")

    provenance = _provenance(config, proteomes, reports, table, threads)
    with (outdir / "provenance.json").open("w") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")

    log(f"\nTotal runtime {(time.monotonic() - started) / 60:.1f} min")
    log(f"Consensus table: {output_tsv}")
    _log_summary(summary, log)

    return {
        "table": table,
        "summary": summary,
        "reports": reports,
        "output": output_tsv,
        "collapsed": collapsed_path,
        "figure": figure_path,
        "provenance": provenance,
    }


def _ingest_oma(
    config: RunConfig,
    proteomes: tuple[Proteome, ...],
    outdir: Path,
    log: Callable[[str], None],
) -> tuple[canonical.PredictorResult | None, StageReport]:
    """Parse a user-supplied OMA standalone run; never launch OMA."""
    adapter = OMA()
    source = Path(config.oma).expanduser().resolve()
    pairs_json = outdir / "pairs" / "oma.json"
    log(f"\n[oma] reading user-supplied run at {source}")

    if config.resume and pairs_json.exists():
        result = canonical.read_json(pairs_json)
        log(f"  reusing {pairs_json.name} ({len(result.pairs)} pairs)")
        return result, StageReport(
            tool=adapter.name,
            label=adapter.label,
            status="reused",
            seconds=0.0,
            n_pairs=len(result.pairs),
            method={},
            commit="user-supplied",
        )

    started = time.monotonic()
    try:
        result = adapter.parse(source, proteomes)
    except Exception as error:  # noqa: BLE001 - OMA must not sink the other tools
        elapsed = time.monotonic() - started
        log(f"  FAILED: {error}")
        return None, StageReport(
            tool=adapter.name,
            label=adapter.label,
            status="failed",
            seconds=elapsed,
            n_pairs=0,
            method=adapter.method_profile(
                StageContext(input_dir=source, work_dir=source, proteomes=proteomes)
            ).as_dict(),
            commit="user-supplied",
            error=str(error),
        )

    canonical.write_json(result, pairs_json)
    canonical.write_legacy_json(result, outdir / "pairs" / "oma_result_dict.json")
    elapsed = time.monotonic() - started
    log(f"  {result.summary()} in {elapsed:.1f} s")
    return result, StageReport(
        tool=adapter.name,
        label=adapter.label,
        status="ok",
        seconds=elapsed,
        n_pairs=len(result.pairs),
        method=adapter.method_profile(
            StageContext(input_dir=source, work_dir=source, proteomes=proteomes)
        ).as_dict(),
        commit="user-supplied",
        dropped_unknown=result.dropped_unknown,
        dropped_same_species=result.dropped_same_species,
        note=result.note,
    )


def _preflight(
    adapters: list[Adapter],
    tool_options: dict[str, list[str]],
    log: Callable[[str], None],
) -> None:
    """Fail before the first long run rather than after it.

    Checked against the options this run will actually use, so selecting a
    tree method whose binary is absent is caught now rather than hours in.
    """
    log("\nChecking predictors")
    problems = []
    for adapter in adapters:
        ready, detail = adapter.check(tool_options.get(adapter.name, []))
        log(f"  {adapter.name:<15} {'ok' if ready else 'NOT READY'}  {detail}")
        if not ready:
            problems.append(f"{adapter.name}: {detail}")
    if problems:
        raise PipelineError(
            "predictors are not ready:\n  "
            + "\n  ".join(problems)
            + "\n\nRun: suppOrth install"
        )


def _commit(adapter: Adapter) -> str:
    record = manifest.predictor(adapter.name)
    return (record or {}).get("commit", "unknown")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(
    config: RunConfig,
    proteomes: tuple[Proteome, ...],
    reports: list[StageReport],
    table: pd.DataFrame,
    threads: int,
) -> dict:
    installed = manifest.load()
    return {
        "environment": manifest.environment(),
        "inputs": {
            proteome.label: {
                "path": str(proteome.fasta),
                "n_proteins": proteome.size,
                "sha256": _sha256(proteome.fasta),
            }
            for proteome in proteomes
        },
        "settings": {
            "tools": config.tools,
            "threads": threads,
            "annotate": config.annotate,
            "support_lv": config.support_lv,
            "identity_filter": config.identity_filter,
            "oma": str(config.oma) if config.oma else None,
        },
        "predictors": [
            {
                "tool": report.tool,
                "label": report.label,
                "status": report.status,
                "commit": report.commit,
                "ref": (installed["predictors"].get(report.tool) or {}).get("ref", ""),
                "minutes": round(report.seconds / 60, 2),
                "n_pairs": report.n_pairs,
                "method": report.method,
                "dropped_unknown": report.dropped_unknown,
                "dropped_same_species": report.dropped_same_species,
                "note": report.note,
                "error": report.error,
            }
            for report in reports
        ],
        "shared_binaries": installed["binaries"],
        "result": {
            "n_pairs": int(len(table)),
            "by_support": {
                str(level): int(count)
                for level, count in table["Lv_support"].value_counts().sort_index().items()
            },
        },
    }


def _log_summary(summary: pd.DataFrame, log: Callable[[str], None]) -> None:
    log("\nAgreement between predictors:")
    for _, row in summary.iterrows():
        log(f"  {str(row['SupportedBy']):<40} {row['n_pairs']:>8} pairs")
