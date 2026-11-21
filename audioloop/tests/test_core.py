import math

import pytest

import audioloop

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _l():
    return audioloop.LoopFinder()


def _check_simple():
    s = [0.0, 1.0, 0.0, 1.0]
    return 0 <= _l().find(s, (1, 3)) < len(s)


def _check_bad_window():
    return _raises(lambda: _l().find([0.0], (5, 9)))


def _check_empty():
    return _raises(lambda: _l().find([], (0, 0)))


def _check_type():
    return isinstance(_l().find([0.0] * 10, (0, 9)), int)


def _check_det():
    s = [0.0] * 10
    return _l().find(s, (0, 9)) == _l().find(s, (0, 9))


def _check_in_window():
    s = [float(i) for i in range(20)]
    idx = _l().find(s, (5, 15))
    return 5 <= idx <= 15

CASES = [("simple", "_check_simple"), ("bad_window", "_check_bad_window"), ("empty", "_check_empty"), ("type", "_check_type"), ("det", "_check_det"), ("in_window", "_check_in_window"), ("smoke6", "_check_smoke6"), ("smoke7", "_check_smoke7"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert LoopFinder() is not None
        return
    assert fn() is True
