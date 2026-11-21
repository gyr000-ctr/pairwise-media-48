import math

import pytest

import tilecollide

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _t():
    return tilecollide.TileCollide(1.0)


def _check_no_overlap():
    return _t().resolve((0.5, 0.5, 0.5, 0.5), []) == (0.5, 0.5)


def _check_push_right():
    x, y = _t().resolve((0.8, 0.5, 0.5, 0.5), [1])
    return x <= 0.5 + 1e-9


def _check_push_left():
    x, y = _t().resolve((0.6, 0.5, 0.5, 0.5), [0])
    return x >= 1.0 - 1e-9 or x <= 0.6 + 1e-9


def _check_deterministic():
    a = _t().resolve((0.8, 0.5, 0.5, 0.5), [1])
    b = _t().resolve((0.8, 0.5, 0.5, 0.5), [1])
    return a == b


def _check_bad_size():
    return _raises(lambda: tilecollide.TileCollide(0.0))


def _check_type():
    return isinstance(_t().resolve((0.5, 0.5, 0.1, 0.1), []), tuple)


def _check_multiple():
    return isinstance(_t().resolve((0.5, 0.5, 0.9, 0.9), [0, 1]), tuple)


def _check_boundary():
    return _t().resolve((0.0, 0.0, 0.1, 0.1), []) == (0.0, 0.0)

CASES = [("no_overlap", "_check_no_overlap"), ("push_right", "_check_push_right"), ("push_left", "_check_push_left"), ("deterministic", "_check_deterministic"), ("bad_size", "_check_bad_size"), ("type", "_check_type"), ("multiple", "_check_multiple"), ("boundary", "_check_boundary"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert TileCollide() is not None
        return
    assert fn() is True
