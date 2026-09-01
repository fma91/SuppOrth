"""Record of what is installed, so results can state their own provenance.

Unpinned installs make a consensus unreproducible: two users running the same
SuppOrth version a month apart can get different orthologues with nothing in
the output to explain why. Every predictor is therefore installed at an
explicit upstream ref, and the resolved commit is written here and copied into
each run's provenance file.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__, paths


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load() -> dict[str, Any]:
    path = paths.manifest_path()
    if not path.exists():
        return {"supporth_version": __version__, "predictors": {}, "binaries": {}}
    try:
        with path.open() as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"supporth_version": __version__, "predictors": {}, "binaries": {}}
    data.setdefault("predictors", {})
    data.setdefault("binaries", {})
    return data


def save(data: dict[str, Any]) -> None:
    paths.ensure_layout()
    data["supporth_version"] = __version__
    data["updated"] = _now()
    with paths.manifest_path().open("w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def record_predictor(
    name: str,
    *,
    ref: str,
    commit: str,
    source: str,
    root: Path,
    entrypoint: str,
    notes: str = "",
    interpreter: str = "",
) -> None:
    data = load()
    data["predictors"][name] = {
        "ref": ref,
        "commit": commit,
        "source": source,
        "root": str(root),
        "entrypoint": entrypoint,
        "notes": notes,
        "interpreter": interpreter,
        "installed": _now(),
    }
    save(data)


def record_binary(name: str, *, path: Path, version: str, origin: str) -> None:
    data = load()
    data["binaries"][name] = {
        "path": str(path),
        "version": version,
        "origin": origin,
        "installed": _now(),
    }
    save(data)


def predictor(name: str) -> dict[str, Any] | None:
    return load()["predictors"].get(name)


def environment() -> dict[str, Any]:
    return {
        "supporth_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "home": str(paths.home()),
    }
