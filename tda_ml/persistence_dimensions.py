"""Explicit homology-dimension selection for persistence losses and metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

SUPPORTED_HOMOLOGY_DIMENSIONS = (0, 1)


def normalize_homology_dimensions(value: Iterable[int] | None) -> tuple[int, ...]:
    """Validate an explicit subset of the currently supported dimensions."""
    if value is None:
        return SUPPORTED_HOMOLOGY_DIMENSIONS
    dimensions = tuple(int(dim) for dim in value)
    if not dimensions:
        raise ValueError("homology_dimensions must not be empty")
    if len(dimensions) != len(set(dimensions)):
        raise ValueError(f"homology_dimensions contains duplicates: {dimensions}")
    unsupported = [
        dim for dim in dimensions if dim not in SUPPORTED_HOMOLOGY_DIMENSIONS
    ]
    if unsupported:
        raise ValueError(
            f"Unsupported homology dimensions {unsupported}; "
            f"supported={SUPPORTED_HOMOLOGY_DIMENSIONS}"
        )
    return tuple(sorted(dimensions))


def select_persistence_dimensions(
    persistence_info: Sequence[Any],
    dimensions: Iterable[int] | None,
) -> list[Any]:
    """Select persistence records by their declared ``dimension`` field."""
    requested = normalize_homology_dimensions(dimensions)
    by_dimension: dict[int, Any] = {}
    for info in persistence_info:
        if not hasattr(info, "dimension"):
            raise TypeError("Persistence record is missing a 'dimension' attribute")
        dim = int(info.dimension)
        if dim in by_dimension:
            raise ValueError(f"Duplicate persistence record for H{dim}")
        by_dimension[dim] = info
    missing = [dim for dim in requested if dim not in by_dimension]
    if missing:
        raise ValueError(
            f"Persistence output missing requested dimensions {missing}; "
            f"available={sorted(by_dimension)}"
        )
    return [by_dimension[dim] for dim in requested]
