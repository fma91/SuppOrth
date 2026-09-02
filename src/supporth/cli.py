"""Command-line interface.

Three verbs for the work itself:

    suppOrth install      set up the shared binaries and the predictors
    suppOrth check        report what is installed and whether it runs
    suppOrth run          run every predictor on the proteomes, emit a consensus

and two for the installation, which is large enough to need looking after:

    suppOrth audit        find helper binaries duplicated inside the predictors
    suppOrth disk         report disk usage and reclaim leftovers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, adapters, deps, disk, manifest, paths, shell
from .pipeline import PipelineError, RunConfig, run_pipeline
from .proteomes import ProteomeError


def _identity_filter(value: str) -> float:
    try:
        percent = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("identity_filter must be a number") from error
    if percent < 0 or percent > 100:
        raise argparse.ArgumentTypeError("identity_filter must be between 0 and 100")
    return percent


def _key_values(items: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"expected name=value, got: {item}")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _tool_paths(items: list[str] | None) -> dict[str, Path]:
    """Parse ``[TOOL=]PATH`` arguments; a bare path is keyed by the empty string."""
    parsed: dict[str, Path] = {}
    for item in items or []:
        if "=" in item:
            tool, value = item.split("=", 1)
            parsed[tool.strip()] = Path(value.strip()).expanduser()
        else:
            parsed[""] = Path(item.strip()).expanduser()
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="suppOrth",
        description=(
            "Run several orthology predictors on two or more proteomes and unify "
            "their calls into a single support-annotated table."
        ),
    )
    parser.add_argument("--version", action="version", version=f"SuppOrth {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    install = sub.add_parser("install", help="install shared binaries and predictors")
    install.add_argument("--tools", default="all", help="comma-separated predictors, or 'all'")
    install.add_argument("--ref", action="append", metavar="TOOL=REF",
                         help="pin a predictor to an upstream tag, branch or commit")
    install.add_argument("--adopt", action="append", metavar="BIN=PATH",
                         help="use an existing binary instead of installing one")
    install.add_argument("--no-download", action="store_true",
                         help="never fetch from upstream; only adopt what is already present")
    install.add_argument("--binaries", default="",
                         help="additionally install binaries not required by the selected tools")
    install.add_argument("--python", action="append", metavar="[TOOL=]PATH",
                         help="interpreter to build a predictor's virtualenv from; "
                              "without TOOL= it applies to all")
    install.add_argument("--pfam-db", type=Path, metavar="PATH",
                         help="adopt an already-indexed PfamA profile database instead of "
                              "spending hours building one (SonicParanoid only)")
    install.add_argument("--binaries-only", action="store_true",
                         help="set up the shared binary layer and stop")
    install.add_argument("--force", action="store_true", help="re-clone predictors from scratch")
    install.add_argument("--keep-bundled", action="store_true",
                         help="leave predictors' own bundled helper binaries in place")

    check = sub.add_parser("check", help="report installation status")
    check.add_argument("--tools", default="all")

    run = sub.add_parser("run", help="run the predictors and build the consensus")
    run.add_argument("fastas", nargs="+", type=Path, metavar="FASTA")
    run.add_argument("-o", "--outdir", type=Path, required=True)
    run.add_argument("--tools", default="all")
    run.add_argument("--labels", help="comma-separated species labels, one per FASTA (default: file names)")
    run.add_argument("-t", "--threads", type=int, default=0, help="0 uses all cores but one")
    run.add_argument("--annotate", choices=("diamond", "blast", "none"), default="diamond")
    run.add_argument("--resume", action="store_true", help="reuse completed stages in the output directory")
    run.add_argument("--supportLv", dest="support_lv", type=int, default=3,
                     metavar="N", choices=(1, 2, 3, 4),
                     help="keep pairs found by at least N predictors (default: 3)")
    run.add_argument("--identity_filter", type=_identity_filter, default=80.0, metavar="PCT",
                     help="also keep pairs below --supportLv when Identity is strictly "
                          "above this percent (default: 80; range 0-100)")
    run.add_argument("--collapse", metavar="MAP1,MAP2",
                     help="two protein-to-gene maps (json, pickle or two-column tsv)")
    run.add_argument("--oma", type=Path, metavar="PATH",
                     help="existing OMA standalone run (Output/ or PairwiseOrthologs/); "
                          "OMA is never installed or launched")
    run.add_argument("--opt", action="append", metavar="TOOL=ARGS",
                     help="extra arguments passed through to one predictor")

    audit = sub.add_parser(
        "audit",
        help="find helper binaries duplicated inside the predictor installations",
    )
    audit.add_argument("--link", action="store_true",
                       help="replace duplicates with links to the shared binary (reversible)")
    audit.add_argument("--restore", action="store_true",
                       help="undo --link, putting each predictor's own copy back")

    disk_cmd = sub.add_parser("disk", help="report disk usage and reclaim leftovers")
    disk_cmd.add_argument("--clean", action="store_true",
                          help="delete the leftovers listed by a plain 'suppOrth disk'")
    disk_cmd.add_argument("--all", action="store_true",
                          help="with --clean, also delete checkouts of refs that are not "
                               "installed; reinstalling them means cloning again")

    sub.add_parser("tools", help="list available predictors")
    return parser


def cmd_install(args: argparse.Namespace) -> int:
    paths.ensure_layout()
    selected = adapters.resolve(args.tools)
    refs = _key_values(args.ref)
    adopt = {k: Path(v) for k, v in _key_values(args.adopt).items()}

    # Validated before anything is fetched or compiled: a mistyped path should
    # cost seconds, not the rebuild it was meant to replace.
    if args.pfam_db:
        capable = [a for a in selected if hasattr(a, "adopt_pfam_profiles")]
        if not capable:
            print(
                "error: --pfam-db applies to none of the selected predictors",
                file=sys.stderr,
            )
            return 2
        for adapter in capable:
            problem = adapter.pfam_source_problem(args.pfam_db)
            if problem:
                print(f"error: {problem}", file=sys.stderr)
                return 2

    print(f"SuppOrth home: {paths.home()}")
    print(f"Platform: {shell.platform_key()}")

    print("\nShared binaries")
    wanted = deps.binaries_for(a.name for a in selected)
    for extra in (b.strip() for b in args.binaries.split(",") if b.strip()):
        if extra not in deps.SPECS:
            print(f"error: unknown binary {extra!r}; known: {', '.join(sorted(deps.SPECS))}",
                  file=sys.stderr)
            return 2
        if extra not in wanted:
            wanted.append(extra)
    try:
        deps.ensure(
            wanted,
            allow_download=not args.no_download,
            adopt=adopt,
        )
    except deps.DependencyError as error:
        print(f"\nerror: {error}", file=sys.stderr)
        return 1

    if args.binaries_only:
        return 0

    pythons = _tool_paths(args.python)

    for adapter in selected:
        print(f"\n{adapter.name}")
        extras = {}
        # Routed by capability rather than by name, so the flag reaches whichever
        # predictors can use a prebuilt profile database.
        if args.pfam_db and hasattr(adapter, "adopt_pfam_profiles"):
            extras["pfam_db"] = args.pfam_db
        try:
            adapter.install(
                ref=refs.get(adapter.name),
                force=args.force,
                keep_bundled=args.keep_bundled,
                python=pythons.get(adapter.name, pythons.get("")),
                log=print,
                **extras,
            )
        except Exception as error:  # noqa: BLE001 - report and continue to the next tool
            print(f"  install failed: {error}", file=sys.stderr)

    print("\nDone. Verify with: suppOrth check")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    print(f"SuppOrth {__version__}")
    print(f"home: {paths.home()}")

    print("\nShared binaries")
    for status in deps.status():
        location = str(status.path) if status.path else "-"
        print(f"  {status.name:<10} {status.origin:<10} {status.version[:48]:<50} {location}")

    print("\nPredictors")
    ok = True
    for adapter in adapters.resolve(args.tools):
        ready, detail = adapter.check()
        record = manifest.predictor(adapter.name) or {}
        commit = record.get("commit", "")[:10]
        ref = record.get("ref", "")
        print(f"  {adapter.name:<15} {'ok' if ready else 'NOT READY':<10} {ref:<10} {commit:<12} {detail}")
        ok = ok and ready
    return 0 if ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    if len(args.fastas) < 2:
        print("error: need at least two FASTA files", file=sys.stderr)
        return 2

    labels = None
    if args.labels:
        parts = [p.strip() for p in args.labels.split(",")]
        if len(parts) != len(args.fastas):
            print(
                f"error: --labels needs {len(args.fastas)} comma-separated names, "
                f"got {len(parts)}",
                file=sys.stderr,
            )
            return 2
        labels = tuple(parts)

    collapse = None
    if args.collapse:
        if len(args.fastas) != 2:
            print("error: --collapse is only defined for exactly two proteomes", file=sys.stderr)
            return 2
        parts = [Path(p.strip()) for p in args.collapse.split(",")]
        if len(parts) != 2:
            print("error: --collapse needs exactly two files", file=sys.stderr)
            return 2
        collapse = (parts[0], parts[1])

    tool_options: dict[str, list[str]] = {}
    for tool, argument_string in _key_values(args.opt).items():
        tool_options[tool] = argument_string.split()

    config = RunConfig(
        fastas=tuple(args.fastas),
        outdir=args.outdir,
        tools=args.tools,
        labels=labels,
        threads=args.threads,
        annotate=args.annotate,
        resume=args.resume,
        support_lv=args.support_lv,
        identity_filter=args.identity_filter,
        collapse=collapse,
        tool_options=tool_options,
        oma=args.oma,
    )

    try:
        run_pipeline(config)
    except (PipelineError, ProteomeError) as error:
        print(f"\nerror: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted; completed stages can be reused with --resume", file=sys.stderr)
        return 130
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    if args.restore:
        print("Restoring bundled binaries")
        count = deps.restore_bundled()
        print(f"{count} restored")
        return 0

    print(f"Shared bin: {paths.shared_bin()}")
    bundled = deps.find_bundled()
    if not bundled:
        print("\nNo helper binaries found inside the predictor installations.")
        return 0

    print(f"\nFound {len(bundled)} helper binaries inside predictor installations:\n")
    print(f"  {'predictor':<16}{'binary':<12}{'state':<12}{'size':>10}  path")
    for item in bundled:
        size = f"{item.size / 1e6:.1f} MB" if item.size else "-"
        print(f"  {item.owner:<16}{item.name:<12}{item.kind:<12}{size:>10}  {item.path}")

    replaceable = [b for b in bundled if not b.linked]
    if args.link:
        print("\nLinking to the shared binary")
        count = deps.link_bundled(replaceable)
        print(f"{count} replaced; originals kept as *.supporth-orig")
        return 0

    if replaceable:
        print(
            "\nThese are separate copies from the shared bin. 'divergent' means the "
            "\nbundled build differs from the shared one, so the two predictors are "
            "\nnot searching with the same program."
            "\n\nCollapse them onto the shared copy with: suppOrth audit --link"
            "\nThat is reversible, but note a predictor may depend on the exact build "
            "\nit ships, so verify results afterwards."
        )
    return 0


def cmd_disk(args: argparse.Namespace) -> int:
    print(f"SuppOrth home: {paths.home()}")

    entries = disk.usage()
    total = sum(entry.size for entry in entries)
    if not entries:
        print("\nNothing installed yet.")
        return 0

    print(f"\nUsing {disk.human(total)}\n")
    for entry in entries:
        print(f"  {disk.human(entry.size):>10}  {entry.label}")

    items = disk.reclaimable()
    if not args.all:
        items = [item for item in items if item.cost == "nothing"]
    if not items:
        print("\nNothing to reclaim.")
        return 0

    reclaimable_total = sum(item.size for item in items)
    if not args.clean:
        print(f"\nReclaimable: {disk.human(reclaimable_total)}\n")
        for item in items:
            print(f"  {disk.human(item.size):>10}  {item.reason}")
        print("\nDelete these with: suppOrth disk --clean")
        if any(item.cost == "refetch" for item in disk.reclaimable()) and not args.all:
            print("Checkouts of refs you no longer use are excluded; add --all to include them.")
        return 0

    freed = disk.remove(items)
    print(f"\nFreed {disk.human(freed)}")
    return 0


def cmd_tools(_: argparse.Namespace) -> int:
    for name in adapters.DEFAULT_ORDER:
        adapter = adapters.get(name)
        print(f"{adapter.label}  {adapter.name:<15} {adapter.repo} @ {adapter.default_ref}")
    print("O  oma             user-supplied (--oma PATH to an existing standalone run)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "install": cmd_install,
        "check": cmd_check,
        "run": cmd_run,
        "audit": cmd_audit,
        "disk": cmd_disk,
        "tools": cmd_tools,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
