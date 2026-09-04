"""Sensor name -> reader instance."""

from __future__ import annotations

from collections.abc import Callable

from sattsr.io.base import Reader
from sattsr.io.goes import Goes19Reader
from sattsr.io.himawari import HimawariReader
from sattsr.io.insat import InsatReader

READERS: dict[str, Callable[[], Reader]] = {
    "goes19": Goes19Reader,
    "himawari8": HimawariReader,
    "insat3dr": lambda: InsatReader(sensor="insat3dr"),
    "insat3ds": lambda: InsatReader(sensor="insat3ds"),
}


def get_reader(sensor: str) -> Reader:
    """Return a fresh reader for `sensor`, or raise KeyError."""
    try:
        factory = READERS[sensor]
    except KeyError as exc:
        raise KeyError(f"unsupported sensor {sensor!r}; known: {sorted(READERS)}") from exc
    return factory()
