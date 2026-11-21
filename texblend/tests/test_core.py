import math

import pytest

import texblend

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

T = [[0.0, 1.0], [2.0, 3.0]]


def _b():
    return texblend.TexBlend()


def _check_corner():
    return _b().sample(T, 0.0, 0.0) == 0.0


def _check_center():
    return abs(_b().sample(T, 0.5, 0.5) - 1.5) < 1e-9


def _check_clamp():
    return _b().sample(T, 2.0, 2.0) == 3.0


def _check_bad_tex():
    return _raises(lambda: _b().sample([], 0.0, 0.0))


def _check_type():
    return isinstance(_b().sample(T, 0.2, 0.2), float)


def _check_range():
    return 0.0 <= _b().sample(T, 0.3, 0.7) <= 3.0


def _check_det():
    return _b().sample(T, 0.4, 0.4) == _b().sample(T, 0.4, 0.4)


def _check_no_mut():
    tex = [row[:] for row in T]
    _b().sample(T, 0.5, 0.5)
    return T == tex

CASES = [("corner", "_check_corner"), ("center", "_check_center"), ("clamp", "_check_clamp"), ("bad_tex", "_check_bad_tex"), ("type", "_check_type"), ("range", "_check_range"), ("det", "_check_det"), ("no_mut", "_check_no_mut"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert TexBlend() is not None
        return
    assert fn() is True
