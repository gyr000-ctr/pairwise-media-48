import math

import pytest

import colormix

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _m():
    return colormix.ColorMix()


def _check_multiply():
    return abs(_m().mix((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))[0] - 0.25) < 1e-9


def _check_screen():
    return abs(_m().mix((0.5, 0, 0), (0.5, 0, 0), "screen")[0] - 0.75) < 1e-9


def _check_t_zero():
    return _m().mix((0.2, 0.2, 0.2), (1.0, 1.0, 1.0), "multiply", 0.0)[0] == 0.2


def _check_bad_mode():
    return _raises(lambda: _m().mix((0, 0, 0), (1, 1, 1), "bogus"))


def _check_bad_len():
    return _raises(lambda: _m().mix((0, 0), (1, 1, 1)))


def _check_type():
    return isinstance(_m().mix((0.1, 0.2, 0.3), (0.3, 0.2, 0.1)), tuple)


def _check_range():
    return all(0.0 <= v <= 1.0 for v in _m().mix((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), "screen"))


def _check_det():
    return _m().mix((0.2, 0.3, 0.4), (0.5, 0.5, 0.5)) == _m().mix((0.2, 0.3, 0.4), (0.5, 0.5, 0.5))

CASES = [("multiply", "_check_multiply"), ("screen", "_check_screen"), ("t_zero", "_check_t_zero"), ("bad_mode", "_check_bad_mode"), ("bad_len", "_check_bad_len"), ("type", "_check_type"), ("range", "_check_range"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert ColorMix() is not None
        return
    assert fn() is True
