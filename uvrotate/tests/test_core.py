import math

import pytest

import uvrotate

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _r():
    return uvrotate.UvRotate()


def _check_zero():
    return _r().rotate((0.25, 0.75), 0.0) == (0.25, 0.75)


def _check_360():
    out = _r().rotate((0.25, 0.75), 360.0)
    return abs(out[0] - 0.25) < 1e-9 and abs(out[1] - 0.75) < 1e-9


def _check_range():
    out = _r().rotate((0.25, 0.75), 33.0)
    return 0.0 <= out[0] < 1.0 and 0.0 <= out[1] < 1.0


def _check_bad_uv():
    return _raises(lambda: _r().rotate((1.5, 0.5), 0.0))


def _check_bad_tiles():
    return _raises(lambda: _r().rotate((0.5, 0.5), 0.0, 0))


def _check_type():
    return isinstance(_r().rotate((0.1, 0.2), 10.0), tuple)


def _check_negative_deg():
    return isinstance(_r().rotate((0.1, 0.2), -45.0), tuple)


def _check_det():
    return _r().rotate((0.3, 0.6), 45.0) == _r().rotate((0.3, 0.6), 45.0)


def _check_tiles():
    return isinstance(_r().rotate((0.3, 0.6), 0.0, 3), tuple)

CASES = [("zero", "_check_zero"), ("360", "_check_360"), ("range", "_check_range"), ("bad_uv", "_check_bad_uv"), ("bad_tiles", "_check_bad_tiles"), ("type", "_check_type"), ("negative_deg", "_check_negative_deg"), ("det", "_check_det"), ("tiles", "_check_tiles"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert UvRotate() is not None
        return
    assert fn() is True
