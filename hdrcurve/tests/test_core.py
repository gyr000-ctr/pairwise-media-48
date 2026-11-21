import math

import pytest

import hdrcurve

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _c():
    return hdrcurve.ToneCurve(2.2, 256)


def _check_zero():
    return _c().lookup(0.0) == 0.0


def _check_one():
    return abs(_c().lookup(1.0) - 1.0) < 1e-9


def _check_monotonic():
    return _c().lookup(0.2) <= _c().lookup(0.8)


def _check_bad_gamma():
    return _raises(lambda: hdrcurve.ToneCurve(0.0))


def _check_bad_levels():
    return _raises(lambda: hdrcurve.ToneCurve(2.2, 0))


def _check_clamp():
    return _c().lookup(2.0) == 1.0


def _check_type():
    return isinstance(_c().lookup(0.5), float)


def _check_range():
    return 0.0 <= _c().lookup(0.3) <= 1.0


def _check_det():
    return _c().lookup(0.4) == _c().lookup(0.4)

CASES = [("zero", "_check_zero"), ("one", "_check_one"), ("monotonic", "_check_monotonic"), ("bad_gamma", "_check_bad_gamma"), ("bad_levels", "_check_bad_levels"), ("clamp", "_check_clamp"), ("type", "_check_type"), ("range", "_check_range"), ("det", "_check_det"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert ToneCurve() is not None
        return
    assert fn() is True
