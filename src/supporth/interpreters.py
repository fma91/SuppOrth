"""Choosing a Python interpreter for a predictor's private environment.

SuppOrth cannot assume the interpreter it runs on suits every predictor:
SonicParanoid 2.0.9 requires Python >=3.10,<3.13, while macOS still ships 3.9.
Each adapter therefore declares what it needs, and an interpreter satisfying
that is located before its virtualenv is built.

Only the *interpreter* is borrowed. Packages are always installed with pip into
a virtualenv SuppOrth owns, so an existing environment is never modified.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from . import shell

Version = tuple[int, int]

_CONSTRAINT = re.compile(r"(>=|<=|==|>|<)\s*(\d+)\.(\d+)")

# Interpreters are looked for in these roots when none on PATH fits. Conda and
# pyenv installations are read as a source of interpreters only.
ENV_ROOTS = (
    "/opt/anaconda3/envs",
    "/opt/miniconda3/envs",
    "~/anaconda3/envs",
    "~/miniconda3/envs",
    "~/miniforge3/envs",
    "~/mambaforge/envs",
    "~/.pyenv/versions",
)

SEARCH_MINOR_RANGE = range(14, 8, -1)  # 3.13 down to 3.9, newest first


class InterpreterError(RuntimeError):
    pass


@dataclass(frozen=True)
class Constraint:
    """A parsed ``python_requires`` string."""

    raw: str
    minimum: Version | None = None
    minimum_inclusive: bool = True
    maximum: Version | None = None
    maximum_inclusive: bool = False

    def allows(self, version: Version) -> bool:
        if self.minimum is not None:
            if version < self.minimum:
                return False
            if version == self.minimum and not self.minimum_inclusive:
                return False
        if self.maximum is not None:
            if version > self.maximum:
                return False
            if version == self.maximum and not self.maximum_inclusive:
                return False
        return True

    def describe(self) -> str:
        return self.raw or "any version"


def parse_requires(spec: str | None) -> Constraint:
    if not spec:
        return Constraint(raw="")
    minimum = maximum = None
    minimum_inclusive = True
    maximum_inclusive = False

    for operator, major, minor in _CONSTRAINT.findall(spec):
        version = (int(major), int(minor))
        if operator in {">=", ">"}:
            minimum = version
            minimum_inclusive = operator == ">="
        elif operator in {"<=", "<"}:
            maximum = version
            maximum_inclusive = operator == "<="
        elif operator == "==":
            minimum = maximum = version
            minimum_inclusive = maximum_inclusive = True

    return Constraint(
        raw=spec,
        minimum=minimum,
        minimum_inclusive=minimum_inclusive,
        maximum=maximum,
        maximum_inclusive=maximum_inclusive,
    )


def version_of(interpreter: Path | str) -> Version | None:
    """Ask an interpreter for its own version, rather than guessing from its name."""
    try:
        result = shell.run(
            [str(interpreter), "-c", "import sys; print(sys.version_info[0], sys.version_info[1])"],
            check=False,
            timeout=60,
        )
    except OSError:
        return None
    fields = (result.stdout or "").split()
    if len(fields) < 2:
        return None
    try:
        return (int(fields[0]), int(fields[1]))
    except ValueError:
        return None


# Directory names conda-style distributions install under. SonicParanoid keys
# its behaviour off exactly this: seeing one of these in sys.exec_prefix, it
# takes its helper binaries from PATH instead of demanding exact versions
# inside its own package directory.
CONDA_MARKERS = (
    "conda",
    "anaconda",
    "anaconda2",
    "anaconda3",
    "miniconda",
    "miniconda2",
    "miniconda3",
    "miniforge",
    "miniforge3",
    "mambaforge",
    "mambaforge3",
)


def is_conda_style(interpreter: Path | str) -> bool:
    """Whether an interpreter belongs to a conda-style distribution."""
    parts = Path(interpreter).resolve().parts
    return any(marker in parts for marker in CONDA_MARKERS)


def candidates() -> Iterator[Path]:
    """Plausible interpreters, most conventional first."""
    yield Path(sys.executable)

    for minor in SEARCH_MINOR_RANGE:
        found = shell.which(f"python3.{minor}")
        if found:
            yield found

    for root in ENV_ROOTS:
        base = Path(root).expanduser()
        if not base.is_dir():
            continue
        try:
            entries = sorted(base.iterdir(), reverse=True)
        except OSError:
            continue
        for entry in entries:
            interpreter = entry / "bin" / "python3"
            if not interpreter.exists():
                interpreter = entry / "bin" / "python"
            if interpreter.exists():
                yield interpreter


def find(
    spec: str | None,
    *,
    explicit: Path | str | None = None,
    prefer_conda: bool = False,
) -> Path:
    """An interpreter satisfying ``spec``.

    ``explicit`` is honoured if given and suitable, so a user can always
    override the search. ``prefer_conda`` picks a conda-style interpreter among
    the suitable ones when there is a choice, for predictors that behave
    differently depending on where their interpreter came from.
    """
    constraint = parse_requires(spec)

    if explicit:
        path = Path(explicit).expanduser()
        version = version_of(path)
        if version is None:
            raise InterpreterError(f"{path} is not a usable Python interpreter")
        if not constraint.allows(version):
            raise InterpreterError(
                f"{path} is Python {version[0]}.{version[1]}, "
                f"which does not satisfy {constraint.describe()}"
            )
        return path

    seen: set[Path] = set()
    inspected: list[str] = []
    suitable: list[Path] = []
    for candidate in candidates():
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        version = version_of(candidate)
        if version is None:
            continue
        if constraint.allows(version):
            if not prefer_conda:
                return candidate
            suitable.append(candidate)
            if is_conda_style(candidate):
                return candidate
        else:
            inspected.append(f"{version[0]}.{version[1]} at {candidate}")

    if suitable:
        return suitable[0]

    raise InterpreterError(
        f"no Python interpreter satisfying {constraint.describe()} was found.\n"
        f"  inspected: {', '.join(inspected) or 'none'}\n"
        f"  provide one with: suppOrth install --python /path/to/python"
    )


def describe(interpreter: Path | str) -> str:
    version = version_of(interpreter)
    label = f"{version[0]}.{version[1]}" if version else "unknown"
    return f"Python {label} ({interpreter})"
