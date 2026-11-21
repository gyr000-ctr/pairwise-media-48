import math

import pytest

import texmip

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def _m():
    return texmip.TexMip(8, 4)


def _check_level0():
    return _m().level(1.0) == 0


def _check_level_small():
    return _m().level(0.01) >= 0


def _check_clamp_max():
    return _m().level(1e-6) <= 3


def _check_bad_scale():
    return _raises(lambda: _m().level(0.0))


def _check_type():
    return isinstance(_m().level(0.5), int)


def _check_size():
    return _m().size_at(1) == (4, 2)


def _check_bad_level():
    return _raises(lambda: _m().size_at(9))


def _check_det():
    return _m().level(0.3) == _m().level(0.3)

CASES = [("level0", "_check_level0"), ("level_small", "_check_level_small"), ("clamp_max", "_check_clamp_max"), ("bad_scale", "_check_bad_scale"), ("type", "_check_type"), ("size", "_check_size"), ("bad_level", "_check_bad_level"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert TexMip() is not None
        return
    assert fn() is True
