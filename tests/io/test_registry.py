from __future__ import annotations

import pytest

from sattsr.io.goes import Goes19Reader
from sattsr.io.himawari import HimawariReader
from sattsr.io.insat import InsatReader
from sattsr.io.registry import get_reader


def test_registry_returns_the_right_reader():
    assert isinstance(get_reader("goes19"), Goes19Reader)
    assert isinstance(get_reader("himawari8"), HimawariReader)
    insat = get_reader("insat3ds")
    assert isinstance(insat, InsatReader)
    assert insat.sensor == "insat3ds"
    assert get_reader("insat3dr").sensor == "insat3dr"


def test_unknown_sensor_raises():
    with pytest.raises(KeyError):
        get_reader("sentinel2")
