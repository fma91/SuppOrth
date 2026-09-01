"""Predictor adapters: one module per orthology tool."""

from __future__ import annotations

from .base import Adapter, MethodProfile, StageContext
from .broccoli import Broccoli
from .oma import OMA
from .orthofinder import OrthoFinder
from .sonicparanoid import SonicParanoid

REGISTRY: dict[str, type[Adapter]] = {
    OrthoFinder.name: OrthoFinder,
    SonicParanoid.name: SonicParanoid,
    Broccoli.name: Broccoli,
    OMA.name: OMA,
}

# Predictors SuppOrth installs and launches. OMA is not among them: the user
# supplies an existing standalone run with --oma.
DEFAULT_ORDER = (OrthoFinder.name, SonicParanoid.name, Broccoli.name)


def get(name: str) -> Adapter:
    key = name.strip().lower()
    if key not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown predictor {name!r}; available: {known}")
    return REGISTRY[key]()


def resolve(names: str | None) -> list[Adapter]:
    """Adapters for a comma-separated selection, kept in run order."""
    if not names or names.strip().lower() == "all":
        return [get(n) for n in DEFAULT_ORDER]
    requested = {n.strip().lower() for n in names.split(",") if n.strip()}
    if "oma" in requested:
        raise KeyError(
            "OMA is not run by suppOrth; pass an existing standalone run with --oma PATH"
        )
    unknown = requested - set(REGISTRY)
    if unknown:
        raise KeyError(
            f"unknown predictor(s): {', '.join(sorted(unknown))}; "
            f"available: {', '.join(sorted(REGISTRY))}"
        )
    return [get(n) for n in DEFAULT_ORDER if n in requested]


__all__ = [
    "Adapter",
    "MethodProfile",
    "StageContext",
    "REGISTRY",
    "DEFAULT_ORDER",
    "get",
    "resolve",
]
