"""The shared external-binary layer.

The predictors depend on an overlapping set of helper programs (DIAMOND,
MMseqs2, BLAST+, IQ-TREE, FastTree, FastME, MCL). Installed independently that
yields several copies at several versions, which is both wasteful and the usual
source of "it worked last month" breakage.

SuppOrth keeps exactly one copy of each in ``<home>/bin`` and puts that
directory first on ``PATH`` for every predictor. Sharing the *installation* is
deliberately not the same as sharing the *choice*: each predictor is still
configured with its own search program, tree builder and clustering method, and
that spread is what makes agreement between them informative.

A binary reaches the shared directory in one of three ways, tried in order:

1. already there and runnable;
2. adopted from the ambient ``PATH`` by symlink (preferred: it is the version
   the user already trusts);
3. fetched from upstream and, where needed, compiled.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

from . import manifest, paths, shell


@dataclass(frozen=True)
class BinarySpec:
    name: str
    purpose: str
    wanted_by: tuple[str, ...]
    version_args: tuple[str, ...] = ("--version",)
    # Upstream asset per platform key; absent platform means "no automatic install".
    downloads: dict[str, str] = field(default_factory=dict)
    # Path of the binary inside the extracted archive, as a glob.
    archive_member: str = ""
    build: str = "archive"  # archive | cc | autotools
    source_url: str = ""
    optional: bool = True
    hint: str = ""
    # Other names the same program is published under; adopted under `name`
    # so tools that hardcode one spelling still find it.
    aliases: tuple[str, ...] = ()
    # Only installed when a run actually asks for it, rather than by default.
    on_demand: bool = False


DIAMOND_VERSION = "2.1.9"
MMSEQS_VERSION = "15-6f452"
MCL_VERSION = "14-137"
BLAST_VERSION = "2.16.0"
IQTREE_VERSION = "2.3.6"

SPECS: dict[str, BinarySpec] = {
    "diamond": BinarySpec(
        name="diamond",
        purpose="fast protein similarity search",
        wanted_by=("broccoli", "supporth"),
        version_args=("--version",),
        downloads={
            "linux-x86_64": f"https://github.com/bbuchfink/diamond/releases/download/v{DIAMOND_VERSION}/diamond-linux64.tar.gz",
            "macos-arm64": f"https://github.com/bbuchfink/diamond/releases/download/v{DIAMOND_VERSION}/diamond-macos.tar.gz",
            "macos-x86_64": f"https://github.com/bbuchfink/diamond/releases/download/v{DIAMOND_VERSION}/diamond-macos.tar.gz",
        },
        archive_member="**/diamond",
        build="archive",
        optional=False,
        hint="required by Broccoli's default search and by consensus annotation",
    ),
    "mmseqs": BinarySpec(
        name="mmseqs",
        purpose="sensitive/fast search used by SonicParanoid",
        wanted_by=("sonicparanoid",),
        version_args=("version",),
        downloads={
            "linux-x86_64": f"https://github.com/soedinglab/MMseqs2/releases/download/{MMSEQS_VERSION}/mmseqs-linux-avx2.tar.gz",
            "linux-arm64": f"https://github.com/soedinglab/MMseqs2/releases/download/{MMSEQS_VERSION}/mmseqs-linux-arm64.tar.gz",
            "macos-arm64": f"https://github.com/soedinglab/MMseqs2/releases/download/{MMSEQS_VERSION}/mmseqs-osx-universal.tar.gz",
            "macos-x86_64": f"https://github.com/soedinglab/MMseqs2/releases/download/{MMSEQS_VERSION}/mmseqs-osx-universal.tar.gz",
        },
        archive_member="**/bin/mmseqs",
        build="archive",
        hint="SonicParanoid ships its own copy; adopting one shared build keeps versions aligned",
    ),
    "FastTree": BinarySpec(
        name="FastTree",
        purpose="approximate maximum-likelihood trees",
        wanted_by=("broccoli",),
        version_args=("-help",),
        source_url="http://www.microbesonline.org/fasttree/FastTree.c",
        build="cc",
        hint="Broccoli's tree builder; OrthoFinder uses it only with -T fasttree",
    ),
    "mcl": BinarySpec(
        name="mcl",
        purpose="Markov clustering of the orthogroup graph",
        wanted_by=("orthofinder",),
        version_args=("--version",),
        source_url=f"https://micans.org/mcl/src/mcl-{MCL_VERSION}.tar.gz",
        build="autotools",
        optional=False,
        hint="required by OrthoFinder",
    ),
    "fastme": BinarySpec(
        name="fastme",
        purpose="distance-based trees (OrthoFinder dendroblast / bundled fallback)",
        wanted_by=("orthofinder",),
        version_args=("--version",),
        hint="distance-based trees; OrthoFinder's bundled copy is used unless one is on PATH",
    ),
    "iqtree": BinarySpec(
        name="iqtree",
        purpose="maximum-likelihood gene trees with model selection",
        wanted_by=("orthofinder",),
        version_args=("--version",),
        downloads={
            "linux-x86_64": f"https://github.com/iqtree/iqtree2/releases/download/v{IQTREE_VERSION}/iqtree-{IQTREE_VERSION}-Linux-intel.tar.gz",
            "macos-arm64": f"https://github.com/iqtree/iqtree2/releases/download/v{IQTREE_VERSION}/iqtree-{IQTREE_VERSION}-macOS.zip",
            "macos-x86_64": f"https://github.com/iqtree/iqtree2/releases/download/v{IQTREE_VERSION}/iqtree-{IQTREE_VERSION}-macOS.zip",
        },
        archive_member="**/bin/iqtree*",
        build="archive",
        aliases=("iqtree2", "iqtree3"),
        on_demand=False,
        hint="OrthoFinder's default gene-tree method (-M msa -T iqtree)",
    ),
    "mafft": BinarySpec(
        name="mafft",
        purpose="multiple sequence alignment for tree inference",
        wanted_by=("orthofinder",),
        version_args=("--version",),
        on_demand=False,
        hint=(
            "OrthoFinder's default MSA stack; adopt-only because the mafft "
            "launcher resolves its own library directory, so copying the binary "
            "alone breaks it. Install via your package manager and re-run "
            "suppOrth install"
        ),
    ),
    "blastp": BinarySpec(
        name="blastp",
        purpose="OrthoFinder search and optional consensus annotation",
        wanted_by=("orthofinder", "sonicparanoid", "supporth"),
        version_args=("-version",),
        downloads={
            "linux-x86_64": f"https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/{BLAST_VERSION}/ncbi-blast-{BLAST_VERSION}+-x64-linux.tar.gz",
            "macos-x86_64": f"https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/{BLAST_VERSION}/ncbi-blast-{BLAST_VERSION}+-x64-macosx.tar.gz",
        },
        archive_member="**/bin/blastp",
        build="archive",
        hint="OrthoFinder's default search; also used with --annotate blast",
    ),
    "makeblastdb": BinarySpec(
        name="makeblastdb",
        purpose="database construction for the blast annotator",
        wanted_by=("orthofinder", "supporth"),
        version_args=("-version",),
        downloads={
            "linux-x86_64": f"https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/{BLAST_VERSION}/ncbi-blast-{BLAST_VERSION}+-x64-linux.tar.gz",
            "macos-x86_64": f"https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/{BLAST_VERSION}/ncbi-blast-{BLAST_VERSION}+-x64-macosx.tar.gz",
        },
        archive_member="**/bin/makeblastdb",
        build="archive",
        hint="paired with blastp for OrthoFinder and --annotate blast",
    ),
}


class DependencyError(RuntimeError):
    pass


@dataclass
class Status:
    name: str
    present: bool
    path: Path | None
    origin: str
    version: str


@dataclass
class Bundled:
    """A helper binary shipped inside a predictor's own installation."""

    name: str
    path: Path
    owner: str
    size: int
    digest: str
    identical_to_shared: bool
    linked: bool
    runnable: bool | None = None

    @property
    def kind(self) -> str:
        if self.linked:
            return "linked"
        if self.runnable is False:
            return "unrunnable"
        return "duplicate" if self.identical_to_shared else "divergent"


def _can_execute(path: Path) -> bool | None:
    """Whether a bundled binary can run on this machine at all.

    Predictors ship binaries for the platform their author built on. A Linux
    ELF executable inside a source tree unpacked on macOS is not a duplicate of
    the shared copy so much as a trap, especially when the predictor puts its
    own directory first on PATH.
    """
    import subprocess

    try:
        subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            timeout=10,
        )
    except OSError:
        return False
    except subprocess.TimeoutExpired:
        return None
    except Exception:  # noqa: BLE001 - diagnosis must not abort the audit
        return None
    return True


def _digest(path: Path, limit: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(limit))
    return digest.hexdigest()[:16]


def _points_at_shared(path: Path) -> bool:
    """Whether a symlink's immediate target is the shared binary.

    Only the first hop is followed. The shared entries are themselves usually
    symlinks onto a system install, so fully resolving would step straight past
    the shared directory to the real binary and report a linked copy as an
    independent duplicate.
    """
    try:
        target = Path(os.readlink(path))
    except OSError:
        return False
    if not target.is_absolute():
        target = (path.parent / target).resolve(strict=False)
    return target.parent == paths.shared_bin()


def _known_names() -> dict[str, str]:
    """Every filename that identifies a shared binary, mapped to its spec name."""
    names: dict[str, str] = {}
    for spec in SPECS.values():
        names[spec.name] = spec.name
        for alias in spec.aliases:
            names[alias] = spec.name
    return names


def find_bundled(roots: dict[str, Path] | None = None) -> list[Bundled]:
    """Locate helper binaries duplicated inside the predictor installations.

    Each predictor ships or builds its own copies of the same few programs.
    Those copies are what drift out of step with each other, so they are worth
    surfacing even when nothing is obviously broken.
    """
    if roots is None:
        roots = {
            name: Path(record["root"])
            for name, record in manifest.load()["predictors"].items()
            if Path(record["root"]).exists()
        }

    known = _known_names()
    shared_digests = {
        spec_name: _digest(paths.shared_bin() / spec_name)
        for spec_name in SPECS
        if (paths.shared_bin() / spec_name).is_file()
    }

    found: list[Bundled] = []
    for owner, root in roots.items():
        for path in root.rglob("*"):
            if ".git" in path.parts or path.is_dir():
                continue
            spec_name = known.get(path.name)
            if spec_name is None:
                continue
            if path.is_symlink() and _points_at_shared(path):
                found.append(Bundled(spec_name, path, owner, 0, "", True, True))
                continue
            if not os.access(path, os.X_OK):
                continue
            try:
                digest = _digest(path)
                size = path.stat().st_size
            except OSError:
                continue
            found.append(
                Bundled(
                    name=spec_name,
                    path=path,
                    owner=owner,
                    size=size,
                    digest=digest,
                    identical_to_shared=shared_digests.get(spec_name) == digest,
                    linked=False,
                    runnable=_can_execute(path),
                )
            )
    return sorted(found, key=lambda b: (b.owner, b.name, str(b.path)))


def link_bundled(items: Sequence[Bundled], *, log: Callable[[str], None] = print) -> int:
    """Point bundled copies at the shared binary, keeping the original next to it.

    This is opt-in on purpose. A predictor may depend on the exact build it
    ships — SonicParanoid in particular tracks MMseqs2 closely — so replacing a
    bundled copy can change results or break the tool. Every replacement leaves
    a ``.supporth-orig`` beside it, and ``suppOrth audit --restore`` puts them
    back.
    """
    replaced = 0
    for item in items:
        if item.linked:
            continue
        shared = paths.shared_bin() / item.name
        if not shared.exists():
            log(f"  {item.name:<10} no shared copy to link to; skipped")
            continue
        backup = item.path.with_name(item.path.name + ".supporth-orig")
        try:
            if not backup.exists():
                item.path.rename(backup)
            elif item.path.exists():
                item.path.unlink()
            item.path.symlink_to(shared)
        except OSError as error:
            log(f"  {item.name:<10} could not link {item.path}: {error}")
            continue
        replaced += 1
        log(f"  {item.owner}/{item.path.name} -> {shared}")
    return replaced


def restore_bundled(roots: dict[str, Path] | None = None, *, log: Callable[[str], None] = print) -> int:
    """Undo `link_bundled`, restoring each predictor's own copy."""
    if roots is None:
        roots = {
            name: Path(record["root"])
            for name, record in manifest.load()["predictors"].items()
            if Path(record["root"]).exists()
        }
    restored = 0
    for owner, root in roots.items():
        for backup in root.rglob("*.supporth-orig"):
            original = backup.with_name(backup.name[: -len(".supporth-orig")])
            if original.is_symlink() or original.exists():
                original.unlink()
            backup.rename(original)
            restored += 1
            log(f"  {owner}/{original.name} restored")
    return restored


def binaries_for(predictors: Iterable[str], *, include_on_demand: bool = False) -> list[str]:
    """Binaries wanted by the given predictors, deduplicated.

    On-demand entries (alternative aligners and tree builders) are left out
    unless asked for: they are only needed when a run selects a method that
    uses them, and fetching them by default would slow every install.
    """
    wanted = {p.lower() for p in predictors} | {"supporth"}
    return [
        name
        for name, spec in SPECS.items()
        if wanted & set(spec.wanted_by) and (include_on_demand or not spec.on_demand)
    ]


def find_system(spec: BinarySpec) -> Path | None:
    """Locate a program on the ambient PATH under any of its published names."""
    for candidate in (spec.name, *spec.aliases):
        found = shutil.which(candidate)
        if found:
            return Path(found)
    return None


def status(names: Sequence[str] | None = None) -> list[Status]:
    recorded = manifest.load()["binaries"]
    out: list[Status] = []
    for name in names or list(SPECS):
        spec = SPECS[name]
        local = paths.shared_bin() / name
        if local.exists():
            path: Path | None = local
            origin = recorded.get(name, {}).get("origin", "shared")
        else:
            found = find_system(spec)
            path = found
            origin = "system" if found else "missing"
        version = (
            shell.probe_version(name, spec.version_args) if path else "absent"
        )
        out.append(Status(name, path is not None, path, origin, version))
    return out


def ensure(
    names: Sequence[str],
    *,
    allow_download: bool = True,
    prefer_system: bool = True,
    adopt: dict[str, Path] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Path]:
    """Make each named binary available in the shared bin directory."""
    paths.ensure_layout()
    resolved: dict[str, Path] = {}
    missing: list[tuple[BinarySpec, str]] = []

    for name in names:
        spec = SPECS.get(name)
        if spec is None:
            raise DependencyError(f"unknown dependency: {name}")

        target = paths.shared_bin() / name

        explicit = (adopt or {}).get(name)
        if explicit:
            resolved[name] = _adopt(spec, Path(explicit).expanduser().resolve(), log)
            continue

        present = target.exists()
        if present and _runs(spec, target):
            log(f"  {name:<10} already present ({target})")
            resolved[name] = target
            continue

        if prefer_system:
            found = find_system(spec)
            if found and found.resolve() != target.resolve():
                resolved[name] = _adopt(spec, found.resolve(), log)
                continue

        if allow_download and (spec.downloads or spec.source_url):
            try:
                resolved[name] = _install_upstream(spec, log)
                continue
            except (DependencyError, urllib.error.URLError, OSError) as error:
                log(f"  {name:<10} automatic install failed: {error}")

        # "present but not runnable" is a different problem from "not installed",
        # and pointing at the wrong one sends people looking in the wrong place.
        reason = (
            f"installed at {target} but it did not run"
            if present
            else (spec.hint or spec.purpose)
        )
        missing.append((spec, reason))

    required = [s for s, _ in missing if not s.optional]
    for spec, reason in missing:
        severity = "MISSING" if not spec.optional else "absent"
        log(f"  {spec.name:<10} {severity} - {reason}")

    if required:
        listed = ", ".join(s.name for s in required)
        raise DependencyError(
            f"required binaries unavailable: {listed}. Install them and re-run, "
            f"or point SuppOrth at existing copies with "
            f"--adopt name=/path/to/binary"
        )
    return resolved


def _runs(spec: BinarySpec, path: Path) -> bool:
    if not os.access(path, os.X_OK):
        return False
    try:
        result = shell.run([path, *spec.version_args], check=False, timeout=120)
    except Exception:  # noqa: BLE001 - a broken binary must not abort install
        return False
    return result.returncode in (0, 1)


def _adopt(spec: BinarySpec, source: Path, log: Callable[[str], None]) -> Path:
    """Link an existing trusted binary into the shared directory."""
    if not source.exists():
        raise DependencyError(f"{spec.name}: {source} does not exist")
    target = paths.shared_bin() / spec.name
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)
    version = shell.probe_version(spec.name, spec.version_args)
    manifest.record_binary(spec.name, path=source, version=version, origin="adopted")
    log(f"  {spec.name:<10} adopted from {source}")
    return target


def _install_upstream(spec: BinarySpec, log: Callable[[str], None]) -> Path:
    if spec.build == "archive":
        return _install_archive(spec, log)
    if spec.build == "cc":
        return _install_cc(spec, log)
    if spec.build == "autotools":
        return _install_autotools(spec, log)
    raise DependencyError(f"{spec.name}: unsupported build strategy {spec.build}")


def _download(url: str, destination: Path, log: Callable[[str], None]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    log(f"  {'':<10} fetching {url}")
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    return destination


def _extract(archive: Path, into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    if archive.suffixes[-2:] == [".tar", ".gz"] or archive.suffix in {".tgz", ".gz"}:
        with tarfile.open(archive) as tar:
            tar.extractall(into)  # noqa: S202 - upstream release assets
    elif archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(into)
    else:
        raise DependencyError(f"cannot extract {archive.name}")
    return into


def _install_archive(spec: BinarySpec, log: Callable[[str], None]) -> Path:
    key = shell.platform_key()
    url = spec.downloads.get(key)
    if not url:
        raise DependencyError(f"no prebuilt asset for platform {key}")

    work = paths.cache_root() / spec.name
    archive = _download(url, work / Path(url).name, log)
    extracted = _extract(archive, work / "unpacked")

    candidates = sorted(extracted.glob(spec.archive_member or f"**/{spec.name}"))
    candidates = [c for c in candidates if c.is_file()]
    if not candidates:
        raise DependencyError(
            f"{spec.name} not found inside {archive.name} "
            f"(looked for {spec.archive_member})"
        )

    target = paths.shared_bin() / spec.name
    if target.exists() or target.is_symlink():
        target.unlink()
    shutil.copy2(candidates[0], target)
    target.chmod(0o755)
    version = shell.probe_version(spec.name, spec.version_args)
    manifest.record_binary(spec.name, path=target, version=version, origin=url)
    log(f"  {spec.name:<10} installed from upstream release")
    return target


def _install_cc(spec: BinarySpec, log: Callable[[str], None]) -> Path:
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if compiler is None:
        raise DependencyError("no C compiler found (need cc, gcc or clang)")

    work = paths.cache_root() / spec.name
    source = _download(spec.source_url, work / Path(spec.source_url).name, log)
    target = paths.shared_bin() / spec.name
    shell.run(
        [
            compiler,
            "-O3",
            "-finline-functions",
            "-funroll-loops",
            "-o",
            str(target),
            str(source),
            "-lm",
        ],
        cwd=work,
        log=work / "build.log",
    )
    target.chmod(0o755)
    version = shell.probe_version(spec.name, spec.version_args)
    manifest.record_binary(spec.name, path=target, version=version, origin=spec.source_url)
    log(f"  {spec.name:<10} compiled from source")
    return target


def _install_autotools(spec: BinarySpec, log: Callable[[str], None]) -> Path:
    work = paths.cache_root() / spec.name
    archive = _download(spec.source_url, work / Path(spec.source_url).name, log)
    extracted = _extract(archive, work / "unpacked")

    roots = [p for p in extracted.iterdir() if p.is_dir()]
    if not roots:
        raise DependencyError(f"{spec.name}: unexpected archive layout")
    src = roots[0]
    prefix = paths.home() / "opt" / spec.name

    shell.run(["./configure", f"--prefix={prefix}"], cwd=src, log=work / "configure.log")
    shell.run(["make", f"-j{shell.cpu_count()}"], cwd=src, log=work / "make.log")
    shell.run(["make", "install"], cwd=src, log=work / "install.log")

    built = prefix / "bin" / spec.name
    if not built.exists():
        raise DependencyError(f"{spec.name}: build finished but {built} is absent")

    target = paths.shared_bin() / spec.name
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(built)
    version = shell.probe_version(spec.name, spec.version_args)
    manifest.record_binary(spec.name, path=built, version=version, origin=spec.source_url)
    log(f"  {spec.name:<10} built from source")
    return target
