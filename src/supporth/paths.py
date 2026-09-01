"""Layout of the SuppOrth managed prefix.

Everything SuppOrth installs lives under a single root (``SUPPORTH_HOME``,
default ``~/.supporth``) so that a user can inspect or delete the whole
installation in one place:

    <home>/bin              shared external binaries, one copy for all predictors
    <home>/tools/<name>/<ref>   predictor sources cloned from upstream
    <home>/cache            download and build scratch space
    <home>/manifest.json    what is installed, at which version
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_HOME = "SUPPORTH_HOME"
DEFAULT_HOME = Path.home() / ".supporth"


def home() -> Path:
    override = os.environ.get(ENV_HOME)
    return Path(override).expanduser().resolve() if override else DEFAULT_HOME


def shared_bin() -> Path:
    """Single directory every predictor resolves its external binaries from."""
    return home() / "bin"


def tools_root() -> Path:
    return home() / "tools"


def tool_dir(name: str, ref: str) -> Path:
    """Source tree for one predictor pinned at one upstream ref."""
    return tools_root() / name / _slug(ref)


def cache_root() -> Path:
    return home() / "cache"


def manifest_path() -> Path:
    return home() / "manifest.json"


def ensure_layout() -> None:
    for directory in (home(), shared_bin(), tools_root(), cache_root()):
        directory.mkdir(parents=True, exist_ok=True)


def _slug(ref: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in ref)
