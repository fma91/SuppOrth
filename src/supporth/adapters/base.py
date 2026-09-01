"""The contract every predictor adapter implements.

Four operations, deliberately small, so that adding a predictor is a new module
rather than a change to the pipeline:

``install``  fetch the tool from upstream at a pinned ref and record the commit
``check``    confirm it can actually start, before a long run is attempted
``run``      execute it on a staged input directory
``parse``    convert its native output into the canonical pair set

Installation is from the upstream repository rather than a packaged build,
because packaged builds of these tools are frequently broken or lag behind, and
because fetching at install time keeps differently-licensed sources out of this
repository.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from .. import disk, interpreters, manifest, paths, shell
from ..canonical import PredictorResult
from ..proteomes import Proteome


class AdapterError(RuntimeError):
    pass


class NotInstalled(AdapterError):
    pass


@dataclass
class MethodProfile:
    """The methodological choices a predictor was run with.

    Recorded per run because it is the substance of the consensus: agreement
    between predictors means more when they searched, aligned, built trees and
    clustered differently.
    """

    search: str = "unspecified"
    clustering: str = "unspecified"
    alignment: str = "none"
    tree: str = "none"
    orthology: str = "unspecified"
    extra: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "search": self.search,
            "clustering": self.clustering,
            "alignment": self.alignment,
            "tree": self.tree,
            "orthology": self.orthology,
            **self.extra,
        }


@dataclass
class StageContext:
    """Everything a single predictor stage needs to run."""

    input_dir: Path
    work_dir: Path
    sp1: Proteome
    sp2: Proteome
    threads: int = 1
    options: list[str] = field(default_factory=list)

    @property
    def log(self) -> Path:
        return self.work_dir / "stage.log"


class Adapter(ABC):
    name: str = ""
    label: str = ""
    repo: str = ""
    default_ref: str = ""
    entrypoint_candidates: tuple[str, ...] = ()
    binaries: tuple[str, ...] = ()
    # True when the predictor puts its own bundled bin directory ahead of PATH,
    # which defeats the shared binary layer unless the bundled copies are linked.
    shadows_path: bool = False
    # Interpreter range the predictor supports, when it is narrower than
    # SuppOrth's own. Empty means "whatever SuppOrth runs on will do".
    python_requires: str = ""
    # Set when the predictor changes behaviour depending on whether its
    # interpreter belongs to a conda-style distribution.
    prefers_conda_interpreter: bool = False

    # ---- installation -------------------------------------------------

    def install(
        self,
        *,
        ref: str | None = None,
        force: bool = False,
        keep_bundled: bool = False,
        python: Path | str | None = None,
        log=print,
        **post_install_options,
    ) -> Path:
        ref = ref or self.default_ref
        root = paths.tool_dir(self.name, ref)

        if root.exists() and force:
            shutil.rmtree(root)

        if not root.exists():
            root.parent.mkdir(parents=True, exist_ok=True)
            log(f"  cloning {self.repo} @ {ref}")
            shell.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    ref,
                    self.repo,
                    str(root),
                ],
                log=paths.cache_root() / f"{self.name}-clone.log",
            )
        else:
            log(f"  source already present at {root}")

        commit = self._resolve_commit(root)
        interpreter = self.select_interpreter(python, log=log)
        # The build step must precede entrypoint discovery: tools installed into
        # a private virtualenv only gain their executable once it has run.
        self.post_install(root, interpreter=interpreter, log=log, **post_install_options)
        entrypoint = self.locate_entrypoint(root)

        if self.shadows_path and not keep_bundled:
            self._collapse_bundled(root, log=log)

        freed = disk.prune_build_artifacts(root)
        if freed:
            log(f"  removed the build tree ({disk.human(freed)})")

        manifest.record_predictor(
            self.name,
            ref=ref,
            commit=commit,
            source=self.repo,
            root=root,
            entrypoint=str(entrypoint),
            notes=self.install_notes,
            interpreter=interpreters.describe(self.venv_python(root) or interpreter),
        )
        log(f"  {self.name} ready at commit {commit[:10]}")
        return root

    install_notes: str = ""

    def _collapse_bundled(self, root: Path, *, log=print) -> None:
        """Point this predictor's bundled helpers at the shared binaries.

        Done at install time for predictors that prepend their own bin
        directory to PATH, because for those the shared layer would otherwise
        be ignored. It also disarms binaries built for another platform, which
        source distributions routinely carry.
        """
        from .. import deps

        bundled = [b for b in deps.find_bundled({self.name: root}) if not b.linked]
        if not bundled:
            return
        unrunnable = [b for b in bundled if b.runnable is False]
        if unrunnable:
            names = ", ".join(sorted({b.name for b in unrunnable}))
            log(f"  bundled {names} cannot run on this machine; using the shared copies")
        else:
            log(f"  linking {len(bundled)} bundled binaries to the shared bin")
        deps.link_bundled(bundled, log=lambda message: log(f"  {message.strip()}"))

    def post_install(self, root: Path, *, interpreter: Path | None = None, log=print) -> None:
        """Hook for tools needing a build step or Python dependencies."""

    def leftovers(self, root: Path) -> list[tuple[Path, str]]:
        """Artefacts this predictor abandoned, as ``(path, reason)`` pairs.

        Only the adapter can recognise a half-finished artefact belonging to
        its own tool, so the disk report asks each predictor instead of
        guessing from file names it does not understand.
        """
        return []

    def select_interpreter(self, explicit: Path | str | None = None, *, log=print) -> Path:
        """The interpreter this predictor's private environment is built from."""
        chosen = interpreters.find(
            self.python_requires,
            explicit=explicit,
            prefer_conda=self.prefers_conda_interpreter,
        )
        if self.python_requires or self.prefers_conda_interpreter:
            log(f"  interpreter: {interpreters.describe(chosen)}")
        return chosen

    def install_to_prefix(
        self,
        root: Path,
        interpreter: Path,
        *,
        log=print,
    ) -> Path:
        """Install the predictor into a private directory with ``pip --target``.

        An alternative to a virtualenv for predictors that inspect their own
        interpreter. A virtualenv would make the interpreter look like it came
        from the virtualenv rather than from wherever it was borrowed, which
        changes how some tools decide where to find their helper binaries. Here
        the interpreter keeps its identity and only the packages are private,
        so the borrowed environment is still never written to.
        """
        target = root / "site-packages"
        target.mkdir(parents=True, exist_ok=True)
        log(f"  installing into {target.name}/ with pip (this compiles extensions)")
        shell.run(
            [str(interpreter), "-m", "pip", "install", "--upgrade", "--target", str(target), "."],
            cwd=root,
            log=root / "install.log",
        )
        (root / "interpreter.txt").write_text(f"{interpreter}\n")
        return target

    def prefix_interpreter(self, root: Path) -> Path | None:
        record = root / "interpreter.txt"
        if not record.exists():
            return None
        path = Path(record.read_text().strip())
        return path if path.exists() else None

    def venv_python(self, root: Path) -> Path | None:
        candidate = root / ".venv" / "bin" / "python"
        return candidate if candidate.exists() else None

    def build_venv(
        self,
        root: Path,
        interpreter: Path | None,
        requirements: Sequence[str] = (),
        *,
        editable_source: bool = False,
        log=print,
    ) -> Path:
        """Create the predictor's private virtualenv and pip-install into it.

        Only pip is used, and only inside this virtualenv, so an interpreter
        borrowed from an existing environment is never written to.
        """
        venv = root / ".venv"
        python = interpreter or Path(self.python())
        if not (venv / "bin" / "python").exists():
            log(f"  creating private virtualenv with {python}")
            shell.run([str(python), "-m", "venv", str(venv)])

        pip = venv / "bin" / "pip"
        shell.run(
            [str(pip), "install", "--upgrade", "pip", "wheel", "setuptools"],
            log=root / "install-pip.log",
        )
        if requirements:
            log(f"  installing {', '.join(requirements)}")
            shell.run([str(pip), "install", *requirements], cwd=root, log=root / "install.log")
        if editable_source:
            log("  building the predictor from source (this compiles extensions)")
            shell.run([str(pip), "install", "."], cwd=root, log=root / "install.log")
        return venv / "bin" / "python"

    def _resolve_commit(self, root: Path) -> str:
        try:
            result = shell.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=False)
            return (result.stdout or "").strip() or "unknown"
        except OSError:
            return "unknown"

    def locate_entrypoint(self, root: Path) -> Path:
        for candidate in self.entrypoint_candidates:
            matches = sorted(root.glob(candidate))
            for match in matches:
                if match.is_file():
                    return match
        listing = ", ".join(sorted(p.name for p in root.iterdir())[:12])
        raise AdapterError(
            f"{self.name}: no entrypoint found under {root}. "
            f"Tried {list(self.entrypoint_candidates)}. Top level contains: {listing}. "
            f"The upstream layout may have changed; pin a different --ref."
        )

    # ---- readiness ----------------------------------------------------

    def installed_root(self) -> Path:
        record = manifest.predictor(self.name)
        if not record:
            raise NotInstalled(
                f"{self.name} is not installed; run: suppOrth install --tools {self.name}"
            )
        root = Path(record["root"])
        if not root.exists():
            raise NotInstalled(
                f"{self.name} is recorded at {root} but that path is gone; "
                f"re-run suppOrth install --force"
            )
        return root

    def extra_binaries(self, options: list[str] | None = None) -> tuple[str, ...]:
        """Binaries required only because of the options this run was given.

        Alternative aligners and tree builders are not needed by every
        configuration, so they are resolved from the run's options rather than
        demanded up front.
        """
        return ()

    def check(self, options: list[str] | None = None) -> tuple[bool, str]:
        """Cheap confirmation that the tool starts, without running an analysis."""
        try:
            root = self.installed_root()
        except NotInstalled as error:
            return False, str(error)
        try:
            entrypoint = self.locate_entrypoint(root)
        except AdapterError as error:
            return False, str(error)

        required = (*self.binaries, *self.extra_binaries(options))
        missing = [b for b in dict.fromkeys(required) if shell.which(b) is None]
        if missing:
            return False, (
                f"missing shared binaries: {', '.join(missing)} "
                f"(suppOrth install --binaries {','.join(missing)})"
            )
        return True, f"ok ({entrypoint.name})"

    # ---- execution ----------------------------------------------------

    @abstractmethod
    def run(self, context: StageContext) -> Path:
        """Execute the predictor and return the root of its native output."""

    @abstractmethod
    def parse(self, native_root: Path, sp1: Proteome, sp2: Proteome) -> PredictorResult:
        """Convert native output into canonical oriented pairs."""

    @abstractmethod
    def method_profile(self, context: StageContext) -> MethodProfile:
        """Search, tree and clustering choices used for this stage."""

    def python(self) -> str:
        import sys

        return sys.executable
